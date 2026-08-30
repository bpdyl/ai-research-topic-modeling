"""
Nepal-AI-Research scraping engine — entry point.

Usage:
    python main.py --email you@example.com --target 1000
    python main.py --email you@example.com --smoke
    python main.py --email you@example.com --target 1200 --ss-api-key XXXX --ieee-key XXXX --resume

Strategy:
    1. Query OpenAlex per time-window (server-side filtered on
       authorships.countries:np + AI/ML topic subfields) — this alone should
       supply the bulk of the corpus (OpenAlex reports 4000+ matching works).
    2. If still under target, query OpenAlex again with title/abstract
       keyword searches to catch papers whose topic tags missed AI.
    3. Query Semantic Scholar per AI-keyword (Nepal-constrained), to catch
       anything OpenAlex missed.
    4. Optionally query IEEE Xplore / Springer if API keys are configured.
    5. Merge + dedupe everything (by DOI, then normalized title).
    6. If --resume was passed, merge with the previous JSON export so a
       partial run is additive rather than replacing/dropping old rows.
    7. Save checkpoint + export CSV/JSON.

NOTE ON NEPJOL: NepJOL (nepjol.info) has no public API. It's an OJS-based
journal aggregator, so a dedicated scraper (paginating journal ToCs) is a
separate, fragile piece of work outside this engine's scope. If your
supervisor requires NepJOL coverage specifically, treat it as manual
supplementation of this dataset rather than part of the automated run.
"""

import argparse
import logging
import os
import sys

# Allow `python scraper/main.py` from the repo root, and `python main.py`
# from inside scraper/.
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import config
import pipeline
from ai_relevance import filter_ai_papers
from sources.openalex import fetch_openalex_papers
from sources.semantic_scholar import fetch_semantic_scholar_papers
from sources.ieee_xplore import fetch_ieee_papers
from sources.springer import fetch_springer_papers

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger("main")


def parse_args():
    p = argparse.ArgumentParser(description="Scrape AI research papers with >=1 Nepal-affiliated author")
    p.add_argument("--email", default=config.OPENALEX_EMAIL, help="Your email, for OpenAlex's polite pool")
    p.add_argument("--target", type=int, default=config.TARGET_PAPER_COUNT)
    p.add_argument("--ss-api-key", default=config.SEMANTIC_SCHOLAR_API_KEY)
    p.add_argument("--ieee-key", default=config.IEEE_API_KEY)
    p.add_argument("--springer-key", default=config.SPRINGER_API_KEY)
    p.add_argument("--require-foreign-coauthor", action="store_true", default=config.REQUIRE_FOREIGN_COAUTHOR)
    p.add_argument("--resume", action="store_true", help="Merge with papers already saved in output JSON")
    p.add_argument("--output-dir", default=config.OUTPUT_DIR)
    p.add_argument("--smoke", action="store_true", help="Tiny live run (a handful of papers) to verify API wiring")
    p.add_argument("--max-results", type=int, default=None, help="Cap per OpenAlex window (also used by --smoke)")
    return p.parse_args()


def main():
    args = parse_args()
    config.OUTPUT_DIR = args.output_dir
    os.makedirs(config.OUTPUT_DIR, exist_ok=True)

    per_window_cap = args.max_results
    keyword_cap = None
    ss_cap = None
    if args.smoke:
        per_window_cap = per_window_cap or 5
        keyword_cap = 5
        ss_cap = 5
        args.target = min(args.target, 15)
        logger.info("Smoke mode: capping fetches so you can verify live APIs quickly")

    previous = pipeline.load_previous_export() if args.resume else []
    if previous:
        logger.info("Resume: loaded %d papers from previous export", len(previous))

    all_batches = [previous] if previous else []
    windows = config.TIME_WINDOWS[:1] if args.smoke else config.TIME_WINDOWS

    # 1. OpenAlex — primary, per time-window so you can watch progress
    #    and so a failed window doesn't lose everything already fetched.
    for start, end in windows:
        logger.info("OpenAlex: fetching %d-%d ...", start, end)
        try:
            batch = fetch_openalex_papers(
                year_start=start,
                year_end=end,
                email=args.email,
                max_results=per_window_cap,
                require_foreign_coauthor=args.require_foreign_coauthor,
            )
        except Exception:
            logger.exception("OpenAlex window %d-%d failed, continuing", start, end)
            batch = []
        all_batches.append(batch)
        logger.info(
            "Running total after OpenAlex %d-%d: %d raw records",
            start,
            end,
            sum(len(b) for b in all_batches),
        )

    merged_so_far = pipeline.merge_and_dedupe(*all_batches)
    logger.info("After OpenAlex + dedup: %d unique papers", len(merged_so_far))

    # 2. OpenAlex keyword pass — applied AI whose *primary* topic is medicine,
    #    agriculture, etc. rather than AI/CV. Always run (except smoke): these
    #    papers are AI research even if OpenAlex filed them under another field.
    if not args.smoke:
        logger.info("OpenAlex keyword search for applied AI ...")
        for keyword in config.AI_KEYWORDS:
            try:
                batch = fetch_openalex_papers(
                    year_start=config.YEAR_START,
                    year_end=config.YEAR_END,
                    email=args.email,
                    subfield_ids=[],  # keyword search; don't also require an AI subfield
                    keyword=keyword,
                    max_results=keyword_cap,
                    require_foreign_coauthor=args.require_foreign_coauthor,
                )
            except Exception:
                logger.exception("OpenAlex keyword %r failed, continuing", keyword)
                batch = []
            all_batches.append(batch)
            merged_so_far = pipeline.merge_and_dedupe(*all_batches)
            logger.info("After OpenAlex keyword %r: %d unique papers", keyword, len(merged_so_far))

    # 3. Semantic Scholar — secondary, fills gaps, only if we're still short
    if len(merged_so_far) < args.target:
        logger.info("Still under target (%d/%d) — querying Semantic Scholar ...", len(merged_so_far), args.target)
        try:
            ss_batch = fetch_semantic_scholar_papers(
                api_key=args.ss_api_key,
                require_foreign_coauthor=args.require_foreign_coauthor,
                max_results_per_query=ss_cap,
                enrich_affiliations=bool(args.ss_api_key),
            )
        except Exception:
            logger.exception("Semantic Scholar fetch failed, continuing without it")
            ss_batch = []
        all_batches.append(ss_batch)
        merged_so_far = pipeline.merge_and_dedupe(*all_batches)
        logger.info("After + Semantic Scholar + dedup: %d unique papers", len(merged_so_far))

    # 4. IEEE Xplore — optional, only runs if a key is configured
    if len(merged_so_far) < args.target and args.ieee_key:
        try:
            ieee_batch = fetch_ieee_papers(
                api_key=args.ieee_key,
                require_foreign_coauthor=args.require_foreign_coauthor,
                max_results=20 if args.smoke else 500,
            )
        except Exception:
            logger.exception("IEEE Xplore fetch failed, continuing without it")
            ieee_batch = []
        all_batches.append(ieee_batch)
        merged_so_far = pipeline.merge_and_dedupe(*all_batches)
        logger.info("After + IEEE Xplore + dedup: %d unique papers", len(merged_so_far))

    # 5. Springer — optional, only runs if a key is configured
    if len(merged_so_far) < args.target and args.springer_key:
        try:
            springer_batch = fetch_springer_papers(
                api_key=args.springer_key,
                require_foreign_coauthor=args.require_foreign_coauthor,
                max_results=20 if args.smoke else 500,
            )
        except Exception:
            logger.exception("Springer fetch failed, continuing without it")
            springer_batch = []
        all_batches.append(springer_batch)
        merged_so_far = pipeline.merge_and_dedupe(*all_batches)
        logger.info("After + Springer + dedup: %d unique papers", len(merged_so_far))

    if args.require_foreign_coauthor:
        merged_so_far = pipeline.filter_require_foreign_coauthor(merged_so_far)

    merged_so_far = filter_ai_papers(merged_so_far)

    # 6. Save
    pipeline.export_to_csv(merged_so_far)
    pipeline.export_to_json(merged_so_far)
    pipeline.save_checkpoint(merged_so_far)

    n_abs = sum(1 for p in merged_so_far if (p.get("abstract") or "").strip())
    logger.info("DONE. %d unique papers (%d with abstracts) saved to %s/", len(merged_so_far), n_abs, config.OUTPUT_DIR)
    if len(merged_so_far) < args.target and not args.smoke:
        logger.warning(
            "Under target (%d/%d). The AI-method filter is strict by design; "
            "add an IEEE/Springer API key or a Semantic Scholar key to widen recall.",
            len(merged_so_far), args.target,
        )
        sys.exit(1)


if __name__ == "__main__":
    main()
