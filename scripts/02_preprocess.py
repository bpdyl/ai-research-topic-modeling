"""
Step 02 — Preprocess the frozen corpus into model-ready representations.

Reads `data/processed/corpus_frozen.jsonl` and produces TWO representations:

  LDA / GA-LDA : tokens -> phrases -> Dictionary -> BoW -> TF-IDF
  BERTopic     : cleaned natural sentences (no tokenisation, no lemmatisation)

The split is deliberate. A sentence transformer needs word order, casing and
function words; handing it the LDA token stream would degrade the benchmark
model and rig the central comparison (OQ-015 / RISK-008).

Usage:
    python scripts/02_preprocess.py
    python scripts/02_preprocess.py --config config/default.yaml
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from nrtm.config import load_config, new_run, set_seed                    # noqa: E402
from nrtm.corpus.load import read_jsonl                                   # noqa: E402
from nrtm.preprocessing.clean import clean_text                           # noqa: E402
from nrtm.preprocessing.tokenize import tokenise_documents                # noqa: E402
from nrtm.preprocessing.represent import (                                # noqa: E402
    build_phrases, build_dictionary, build_bow, build_tfidf,
    vocabulary_stats, write_jsonl_rows,
)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Preprocess the frozen corpus")
    p.add_argument("--config", default=None)
    p.add_argument("--dry-run", action="store_true", help="Report only; write nothing")
    return p.parse_args()


def main() -> int:
    args = parse_args()
    cfg = load_config(args.config)
    set_seed(cfg["seed"])
    pre = cfg["preprocessing"]

    run = new_run(cfg, tag="preprocess")
    print(f"Run: {run.run_id}\n")

    # ── Load the frozen corpus ─────────────────────────────────────────
    docs = read_jsonl(cfg.path("frozen_corpus"))
    print(f"Loaded {len(docs)} frozen documents\n")

    # ── Shared cleaning ────────────────────────────────────────────────
    cleaned = [
        clean_text(
            d.text,
            strip_leading_abstract=pre.get("strip_leading_abstract", True),
            strip_copyright=pre.get("strip_copyright", True),
        )
        for d in docs
    ]
    raw_chars = sum(len(d.text) for d in docs)
    clean_chars = sum(len(c) for c in cleaned)
    print("Cleaning:")
    print(f"  characters {raw_chars:,} -> {clean_chars:,} "
          f"({100 * (raw_chars - clean_chars) / max(raw_chars, 1):.1f}% removed)")

    # ── Branch A: BERTopic (natural text) ──────────────────────────────
    bertopic_rows = [
        {"doc_id": d.doc_id, "year": d.year, "time_window": d.time_window, "text": c}
        for d, c in zip(docs, cleaned)
    ]
    print(f"  BERTopic branch: {len(bertopic_rows)} natural-text documents\n")

    # ── Branch B: LDA (tokens) ─────────────────────────────────────────
    print("Tokenising (lowercase, stopwords, lemmatise, acronyms protected)...")
    tokens = tokenise_documents(
        cleaned,
        domain_stopwords=pre.get("domain_stopwords", []),
        protected_acronyms=pre.get("protected_acronyms", []),
        min_token_length=pre.get("min_token_length", 3),
    )
    pre_phrase = vocabulary_stats(tokens)
    print(f"  {pre_phrase['total_tokens']:,} tokens, "
          f"{pre_phrase['unique_types']:,} unique types")

    print("Detecting phrases...")
    tokens, _models = build_phrases(
        tokens,
        bigrams=pre.get("bigrams", True),
        trigrams=pre.get("trigrams", True),
        min_count=pre.get("phrase_min_count", 10),
        threshold=pre.get("phrase_threshold", 20.0),
    )
    post_phrase = vocabulary_stats(tokens)
    n_phrases = sum(1 for d in tokens for t in d if "_" in t)
    print(f"  {post_phrase['unique_types']:,} unique types after phrasing "
          f"({n_phrases:,} phrase occurrences)")

    # ── Dictionary / BoW / TF-IDF ──────────────────────────────────────
    dcfg = pre.get("dictionary", {})
    dictionary, before = build_dictionary(
        tokens,
        no_below=dcfg.get("no_below", 5),
        no_above=dcfg.get("no_above", 0.5),
        keep_n=dcfg.get("keep_n", 50000),
    )
    print(f"\nDictionary: {before:,} types -> {len(dictionary):,} after "
          f"filter_extremes(no_below={dcfg.get('no_below', 5)}, "
          f"no_above={dcfg.get('no_above', 0.5)})")

    bow = build_bow(tokens, dictionary)
    tfidf_model, tfidf = build_tfidf(bow)
    empty_bow = sum(1 for d in bow if not d)
    print(f"BoW: {sum(len(d) for d in bow):,} non-zero entries, "
          f"{empty_bow} empty documents")
    if empty_bow:
        print("  WARNING: empty documents will produce degenerate topic "
              "distributions. Consider lowering dictionary.no_below.")

    final = vocabulary_stats(tokens, dictionary)
    print(f"\nTokens per document: min {final['tokens_per_doc']['min']} / "
          f"median {final['tokens_per_doc']['median']} / "
          f"max {final['tokens_per_doc']['max']} "
          f"(mean {final['tokens_per_doc']['mean']})")

    # Acronym preservation check — this is a proposal commitment, so verify it
    # actually happened rather than assuming.
    protected = [a.lower() for a in pre.get("protected_acronyms", [])]
    surviving = sorted(a for a in protected if a in dictionary.token2id)
    print(f"\nProtected acronyms surviving into the dictionary "
          f"({len(surviving)}/{len(protected)}): {', '.join(surviving)}")

    print(f"\nTop 20 terms: {', '.join(t for t, _ in final['most_common'][:20])}")

    report = {
        "run_id": run.run_id,
        "n_documents": len(docs),
        "cleaning": {"raw_chars": raw_chars, "clean_chars": clean_chars},
        "before_phrases": {k: v for k, v in pre_phrase.items() if k != "most_common"},
        "after_phrases": {k: v for k, v in post_phrase.items() if k != "most_common"},
        "phrase_occurrences": n_phrases,
        "dictionary": {
            "types_before_filter": before,
            "types_after_filter": len(dictionary),
            "no_below": dcfg.get("no_below", 5),
            "no_above": dcfg.get("no_above", 0.5),
        },
        "bow": {"non_zero_entries": sum(len(d) for d in bow), "empty_documents": empty_bow},
        "acronyms": {"configured": protected, "surviving": surviving},
        "final": final,
    }

    if args.dry_run:
        print("\n[dry-run] nothing written")
        run.write_json("preprocess_report.json", report)
        return 0

    # ── Write artefacts ────────────────────────────────────────────────
    from gensim.corpora import MmCorpus

    write_jsonl_rows(
        [{"doc_id": d.doc_id, "year": d.year, "time_window": d.time_window, "tokens": t}
         for d, t in zip(docs, tokens)],
        cfg.path("tokens"),
    )
    write_jsonl_rows(bertopic_rows, cfg.path("bertopic_docs"))
    dictionary.save(str(cfg.path("dictionary")))
    MmCorpus.serialize(str(cfg.path("bow_corpus")), bow)
    MmCorpus.serialize(str(cfg.path("tfidf_corpus")), tfidf)
    with open(cfg.path("preprocess_report"), "w", encoding="utf-8") as fh:
        json.dump(report, fh, indent=2, ensure_ascii=False)
    run.write_json("preprocess_report.json", report)

    root = cfg.path("tokens").parents[2]
    print(f"\nTokens        -> {cfg.path('tokens').relative_to(root)}")
    print(f"BERTopic docs -> {cfg.path('bertopic_docs').relative_to(root)}")
    print(f"Dictionary    -> {cfg.path('dictionary').relative_to(root)}")
    print(f"BoW / TF-IDF  -> {cfg.path('bow_corpus').relative_to(root)} , "
          f"{cfg.path('tfidf_corpus').relative_to(root)}")
    print(f"Run artefacts -> {run.path.relative_to(root)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
