"""
Step 05 — BERTopic benchmark (EXP-003).

The modern embedding-based comparison for RQ4. Consumes the natural-sentence
branch from Phase 4, never the LDA token stream (OQ-015 / RISK-008).

Usage:
    python scripts/05_bertopic.py
    python scripts/05_bertopic.py --no-stability     # skip the 3-seed refit
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from nrtm.config import load_config, new_run, set_seed                        # noqa: E402
from nrtm.models.bertopic_model import (                                      # noqa: E402
    build_embeddings, fit_bertopic, bertopic_top_words,
    bertopic_doc_topic_matrix, vocabulary_overlap,
)
from nrtm.evaluation.coherence import all_coherences                          # noqa: E402
from nrtm.evaluation.diversity import (                                       # noqa: E402
    topic_diversity, mean_pairwise_jaccard, redundant_topic_pairs,
)
from nrtm.evaluation.stability import matched_jaccard                         # noqa: E402


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Fit the BERTopic benchmark")
    p.add_argument("--config", default=None)
    p.add_argument("--no-stability", action="store_true")
    return p.parse_args()


def main() -> int:
    args = parse_args()
    cfg = load_config(args.config)
    seed = cfg["seed"]
    set_seed(seed)
    bcfg, ecfg, pcfg = cfg["bertopic"], cfg["evaluation"], cfg["preprocessing"]
    processes = cfg.get("coherence_processes", 1)

    run = new_run(cfg, tag="bertopic")
    print(f"Run: {run.run_id}\n")

    # ── Inputs ─────────────────────────────────────────────────────────
    from gensim.corpora import Dictionary, MmCorpus

    docs_path = cfg.path("bertopic_docs")
    if not docs_path.exists():
        raise SystemExit(f"Missing {docs_path}\nRun: python scripts/02_preprocess.py")
    rows = [json.loads(l) for l in open(docs_path, encoding="utf-8") if l.strip()]
    documents = [r["text"] for r in rows]

    # The LDA reference corpus, used to score every model identically.
    token_rows = [json.loads(l) for l in open(cfg.path("tokens"), encoding="utf-8") if l.strip()]
    texts = [r["tokens"] for r in token_rows]
    dictionary = Dictionary.load(str(cfg.path("dictionary")))
    bow = list(MmCorpus(str(cfg.path("bow_corpus"))))

    print(f"Documents: {len(documents)} natural-text (BERTopic branch)")
    print(f"Reference: {len(dictionary)} dictionary terms (shared scoring basis)\n")

    # Same stop-word basis as LDA, so the comparison measures modelling rather
    # than preprocessing.
    import nltk
    stops = set(nltk.corpus.stopwords.words("english")) | set(pcfg.get("domain_stopwords", []))

    # ── Embed once ─────────────────────────────────────────────────────
    print(f"Embedding with {bcfg['embedding_model']}...")
    t0 = time.time()
    embeddings = build_embeddings(documents, bcfg["embedding_model"], show_progress=False)
    print(f"  {embeddings.shape} in {time.time()-t0:.0f}s\n")

    # ── Fit ────────────────────────────────────────────────────────────
    print("Fitting BERTopic (UMAP -> HDBSCAN -> c-TF-IDF)...")
    t0 = time.time()
    model, assignments, _probs, info = fit_bertopic(
        documents, embeddings=embeddings,
        embedding_model=bcfg["embedding_model"],
        min_topic_size=bcfg["min_topic_size"],
        umap_kwargs=bcfg.get("umap"), hdbscan_kwargs=bcfg.get("hdbscan"),
        stopwords=sorted(stops), seed=seed,
    )
    print(f"  fitted in {time.time()-t0:.0f}s")
    print(f"  topics found : {info['n_topics']}")
    print(f"  outliers     : {info['n_outliers']} / {info['n_documents']} "
          f"({100*info['outlier_fraction']:.1f}%)")

    topics = bertopic_top_words(model, top_n=ecfg["top_n_words"])

    # ── Fairness check on the shared scoring basis ──────────────────────
    overlap = vocabulary_overlap(topics, dictionary)
    print(f"\n  topic words present in the LDA dictionary: "
          f"{overlap['topic_words_in_dictionary']}/{overlap['topic_words_total']} "
          f"({100*overlap['coverage']:.1f}%)")
    if overlap["coverage"] < 0.6:
        print("  WARNING: low overlap — BERTopic's coherence is computed on a "
              "materially different word set from LDA's. State this caveat.")

    # ── Metrics ────────────────────────────────────────────────────────
    metrics = all_coherences(
        topics, dictionary, texts=texts, corpus=bow,
        measures=tuple(ecfg["coherence_measures"]), processes=processes,
    )
    metrics["diversity"] = topic_diversity(topics, top_n=ecfg["diversity_top_n"])
    metrics["mean_pairwise_jaccard"] = mean_pairwise_jaccard(topics, top_n=ecfg["diversity_top_n"])
    metrics["num_topics"] = info["n_topics"]
    metrics["outlier_fraction"] = info["outlier_fraction"]

    # ── Stability: refit UMAP+HDBSCAN under new seeds, reusing embeddings ──
    if not args.no_stability:
        n_seeds = cfg["ga"]["stability_seeds"]
        stab_seeds = [seed + i for i in range(n_seeds)]
        print(f"\nStability across seeds {stab_seeds} (embeddings reused)...")
        runs = []
        for s in stab_seeds:
            _m, _a, _p, _i = fit_bertopic(
                documents, embeddings=embeddings,
                embedding_model=bcfg["embedding_model"],
                min_topic_size=bcfg["min_topic_size"],
                umap_kwargs=dict(bcfg.get("umap") or {}, random_state=s),
                hdbscan_kwargs=bcfg.get("hdbscan"),
                stopwords=sorted(stops), seed=s,
            )
            runs.append(bertopic_top_words(_m, top_n=ecfg["top_n_words"]))
            print(f"  seed {s}: {_i['n_topics']} topics, "
                  f"{100*_i['outlier_fraction']:.1f}% outliers")
        from itertools import combinations
        pair = [matched_jaccard(a, b, top_n=ecfg["top_n_words"]) for a, b in combinations(runs, 2)]
        valid = [p for p in pair if p == p]
        metrics["stability"] = sum(valid) / len(valid) if valid else float("nan")
        metrics["stability_pairwise"] = [round(p, 4) for p in pair]

    print("\nBERTopic metrics:")
    for k, v in metrics.items():
        print(f"  {k:<24} {v:.4f}" if isinstance(v, float) else f"  {k:<24} {v}")

    redundant = redundant_topic_pairs(topics, top_n=ecfg["top_n_words"], threshold=0.5)
    print(f"\nNear-duplicate topic pairs (>50% overlap): {len(redundant)}")

    print(f"\nTopics (top {ecfg['top_n_words']} words):")
    for i, t in enumerate(topics):
        print(f"  {i:>2}: {', '.join(t)}")

    # ── Persist ────────────────────────────────────────────────────────
    import numpy as np
    np.save(run.file("doc_topics.npy"),
            bertopic_doc_topic_matrix(assignments, info["n_topics"]))
    np.save(run.file("assignments.npy"), np.array(assignments))

    payload = {
        "experiment": "EXP-003",
        "run_id": run.run_id,
        "seed": seed,
        "embedding_model": bcfg["embedding_model"],
        "parameters": {
            "min_topic_size": bcfg["min_topic_size"],
            "umap": bcfg.get("umap"),
            "hdbscan": bcfg.get("hdbscan"),
        },
        "info": info,
        "vocabulary_overlap": overlap,
        "metrics": metrics,
        "topics": topics,
        "redundant_topic_pairs": redundant,
        "doc_ids": [r["doc_id"] for r in rows],
    }
    run.write_json("bertopic.json", payload)

    root = run.path.parents[2]
    print(f"\nRun artefacts -> {run.path.relative_to(root)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
