"""
Step 10 — Interpretability assessment (EXP-007).

Scores the LLM-as-judge ratings and folds them into the comparison table.

This is explicitly NOT the three-human-rater protocol the proposal commits to
(OQ-012). It is one automated rater, and inter-rater agreement is therefore not
computed. If human ratings are collected they supersede this; the blind sheet
from step 06 is still the instrument for that.

Usage:
    python scripts/10_interpretability.py
"""

from __future__ import annotations

import argparse, glob, json, statistics, sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from nrtm.config import load_config, new_run, set_seed        # noqa: E402


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--config", default=None)
    p.add_argument("--ratings", default="data/processed/interpretability_llm_judge.json")
    args = p.parse_args()

    cfg = load_config(args.config)
    set_seed(cfg["seed"])
    run = new_run(cfg, tag="interpretability")
    print(f"Run: {run.run_id}\n")

    root = cfg.path("frozen_corpus").parents[2]
    data = json.loads((root / args.ratings).read_text(encoding="utf-8"))
    prov = data["_provenance"]
    print(f"Rater      : {prov['rater']} ({prov['rater_type']})")
    print(f"Scale      : {prov['scale']}\n")

    summary = {}
    for model, topics in data.items():
        if model.startswith("_"):
            continue
        scores = [v["rating"] for v in topics.values()]
        summary[model] = {
            "n_topics": len(scores),
            "mean": round(statistics.mean(scores), 3),
            "median": statistics.median(scores),
            "stdev": round(statistics.stdev(scores), 3) if len(scores) > 1 else 0.0,
            "min": min(scores), "max": max(scores),
            "n_rated_4_or_5": sum(1 for s in scores if s >= 4),
            "pct_rated_4_or_5": round(100 * sum(1 for s in scores if s >= 4) / len(scores), 1),
            "scores": scores,
        }

    width = max(len(m) for m in summary) + 2
    print(f"{'Model':<{width}}{'n':>5}{'mean':>8}{'median':>8}{'sd':>7}{'>=4':>7}{'% >=4':>8}")
    print("-" * (width + 43))
    for m, s in summary.items():
        print(f"{m:<{width}}{s['n_topics']:>5}{s['mean']:>8.2f}{s['median']:>8}"
              f"{s['stdev']:>7.2f}{s['n_rated_4_or_5']:>7}{s['pct_rated_4_or_5']:>8.1f}")

    best = max(summary, key=lambda m: summary[m]["mean"])
    print(f"\nHighest mean interpretability: {best} ({summary[best]['mean']:.2f})")

    run.write_json("interpretability.json", {
        "experiment": "EXP-007",
        "run_id": run.run_id,
        "provenance": prov,
        "summary": summary,
        "caveat": ("Single LLM rater. Not the 3-human protocol of OQ-012; "
                   "inter-rater agreement not computable and not reported."),
    })
    print(f"\nRun artefacts -> {run.path.relative_to(root)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
