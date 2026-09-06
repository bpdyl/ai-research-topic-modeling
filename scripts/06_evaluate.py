"""
Step 06 — Comparative evaluation (EXP-004).

Assembles the three-way comparison table promised by the proposal, and emits
the human interpretability rating pack that OQ-012 requires.

The interpretability column cannot be computed. It needs three people rating
each topic. This script produces the blind rating sheet; it does not invent the
numbers.

Usage:
    python scripts/06_evaluate.py
"""

from __future__ import annotations

import argparse
import csv
import json
import random
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from nrtm.config import load_config, new_run, set_seed        # noqa: E402

MODELS = [
    ("Standard LDA", "*_lda_baseline", "lda_baseline.json"),
    ("GA-Optimized LDA", "*_ga_lda", "ga_lda.json"),
    ("BERTopic", "*_bertopic", "bertopic.json"),
]

# (key, label, higher_is_better)
METRICS = [
    ("num_topics", "Topics (K)", None),
    ("c_v", "Coherence C_v", True),
    ("c_npmi", "Coherence c_npmi", True),
    ("u_mass", "Coherence u_mass", True),
    ("diversity", "Topic diversity", True),
    ("mean_pairwise_jaccard", "Mean pairwise Jaccard", False),
    ("stability", "Stability (3 seeds)", True),
    ("perplexity", "Perplexity", False),
    ("outlier_fraction", "Unmodelled documents", False),
]


def latest_run(results_dir: Path, pattern: str, filename: str):
    """Most recent run of a given kind. Smoke runs are excluded by pattern."""
    hits = sorted(p for p in results_dir.glob(f"{pattern}/{filename}")
                  if "smoke" not in p.parent.name)
    if not hits:
        return None, None
    with open(hits[-1], encoding="utf-8") as fh:
        return json.load(fh), hits[-1].parent.name


def fmt(value, key: str) -> str:
    if value is None:
        return "—"
    if key in ("num_topics",):
        return f"{int(value)}"
    if key == "outlier_fraction":
        return f"{100*value:.1f}%"
    if key == "perplexity":
        return f"{value:.1f}"
    return f"{value:.4f}"


def main() -> int:
    p = argparse.ArgumentParser(description="Three-way comparative evaluation")
    p.add_argument("--config", default=None)
    args = p.parse_args()

    cfg = load_config(args.config)
    set_seed(cfg["seed"])
    run = new_run(cfg, tag="evaluate")
    print(f"Run: {run.run_id}\n")

    results_dir = cfg.path("results_dir")
    loaded = {}
    for label, pattern, filename in MODELS:
        payload, run_name = latest_run(results_dir, pattern, filename)
        if payload is None:
            print(f"  MISSING: {label} — no run matching {pattern}/{filename}")
            continue
        loaded[label] = {"payload": payload, "run": run_name}
        print(f"  {label:<20} <- {run_name}")

    if len(loaded) < 2:
        raise SystemExit("\nNeed at least two models to compare.")

    # ── Comparison table ───────────────────────────────────────────────
    print("\n" + "=" * 78)
    print("COMPARATIVE EVALUATION")
    print("=" * 78)
    names = list(loaded)
    width = max(len(n) for n in names) + 2

    header = f"{'Metric':<26}" + "".join(f"{n:>{width}}" for n in names)
    print(header)
    print("-" * len(header))

    table_rows = []
    for key, label, higher_better in METRICS:
        values = [loaded[n]["payload"]["metrics"].get(key) for n in names]

        # LDA assigns every document to a topic distribution, so its unmodelled
        # fraction is genuinely 0, not missing. Without this the table marks
        # BERTopic's 17.1% as "best" purely because it is the only value present.
        if key == "outlier_fraction":
            values = [0.0 if (v is None and "LDA" in n) else v
                      for v, n in zip(values, names)]
        if all(v is None for v in values):
            continue
        cells = [fmt(v, key) for v in values]

        # Mark the winner, but only where "better" is defined.
        if higher_better is not None:
            numeric = [(i, v) for i, v in enumerate(values) if isinstance(v, (int, float))]
            if numeric:
                best_i = (max(numeric, key=lambda x: x[1]) if higher_better
                          else min(numeric, key=lambda x: x[1]))[0]
                cells[best_i] = "*" + cells[best_i]
        print(f"{label:<26}" + "".join(f"{c:>{width}}" for c in cells))
        table_rows.append({"metric": label, "key": key,
                           **{n: values[i] for i, n in enumerate(names)}})

    print("-" * len(header))
    print("* = best on that metric.  Interpretability is NOT here: it requires human raters (OQ-012).")

    # ── Honest caveats, computed rather than asserted ──────────────────
    print("\nCaveats that must accompany this table:")
    bt = loaded.get("BERTopic", {}).get("payload")
    if bt:
        of = bt["metrics"].get("outlier_fraction")
        if of:
            print(f"  - BERTopic leaves {100*of:.1f}% of documents unmodelled (HDBSCAN outliers); "
                  "LDA models 100%. Coverage differs, so the coherence figures are not "
                  "computed over the same document set.")
        cov = bt.get("vocabulary_overlap", {}).get("coverage")
        if cov is not None:
            print(f"  - Only {100*cov:.1f}% of BERTopic's topic words exist in the LDA dictionary "
                  "used as the shared scoring basis; the remainder are dropped before "
                  "coherence is computed.")
    ga = loaded.get("GA-Optimized LDA", {}).get("payload")
    if ga:
        print("  - The GA optimised C_v directly. Its C_v advantage over the baseline is therefore "
              "partly expected; the meaningful comparisons are c_npmi, u_mass and perplexity, "
              "which it did not optimise.")

    # ── Human interpretability rating pack (OQ-012) ────────────────────
    rng = random.Random(cfg["seed"])
    sheet = []
    for name in names:
        for i, words in enumerate(loaded[name]["payload"]["topics"]):
            sheet.append({"model": name, "topic_id": i, "top_words": ", ".join(words)})
    rng.shuffle(sheet)   # blind: raters must not see which model produced a topic

    sheet_path = run.file("interpretability_rating_sheet.csv")
    with open(sheet_path, "w", encoding="utf-8", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=["item", "top_words", "rating_1_to_5", "suggested_label",
                                           "_model", "_topic_id"])
        w.writeheader()
        for n, row in enumerate(sheet, 1):
            w.writerow({"item": n, "top_words": row["top_words"], "rating_1_to_5": "",
                        "suggested_label": "", "_model": row["model"], "_topic_id": row["topic_id"]})

    blind_path = run.file("interpretability_rating_sheet_BLIND.csv")
    with open(blind_path, "w", encoding="utf-8", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=["item", "top_words", "rating_1_to_5", "suggested_label"])
        w.writeheader()
        for n, row in enumerate(sheet, 1):
            w.writerow({"item": n, "top_words": row["top_words"],
                        "rating_1_to_5": "", "suggested_label": ""})

    print(f"\nInterpretability rating pack ({len(sheet)} topics, shuffled and blind):")
    print(f"  give raters : {blind_path.name}")
    print(f"  key (do not share until ratings are returned) : {sheet_path.name}")
    print("  3 raters x 1-5 scale; report mean plus inter-rater agreement.")

    payload = {
        "experiment": "EXP-004",
        "run_id": run.run_id,
        "models": {n: loaded[n]["run"] for n in names},
        "table": table_rows,
        "n_topics_to_rate": len(sheet),
        "interpretability_status": "PENDING — requires 3 human raters (OQ-012)",
    }
    run.write_json("comparison.json", payload)

    # Markdown table, ready to paste into the paper.
    md = ["| Metric | " + " | ".join(names) + " |",
          "|---|" + "---|" * len(names)]
    for row in table_rows:
        md.append("| " + row["metric"] + " | " +
                  " | ".join(fmt(row[n], row["key"]) for n in names) + " |")
    md.append("| Interpretability (1-5) | " + " | ".join(["pending"] * len(names)) + " |")
    (run.path / "comparison_table.md").write_text("\n".join(md), encoding="utf-8")

    root = run.path.parents[2]
    print(f"\nRun artefacts -> {run.path.relative_to(root)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
