"""
Step 07 — Thematic evolution (EXP-005).

Implements DEC-013. One global model fitted on the whole corpus; topic
proportions reported per time window, normalised within window.

GA-Optimized LDA is the primary model here — not because it won every metric
(BERTopic beat it on C_v and stability) but because of what the temporal
analysis needs: LDA gives every document a topic *distribution* over 100% of
the corpus, whereas BERTopic gives a hard assignment and leaves 17.1% of
documents unclustered. Proportions computed from a model that ignores a sixth
of the corpus, disproportionately in some windows, would be misleading.
BERTopic is run as a robustness check.

Usage:
    python scripts/07_temporal.py
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from nrtm.config import load_config, new_run, set_seed          # noqa: E402
from nrtm.temporal.proportions import window_proportions, topic_trends   # noqa: E402
from nrtm.viz.figures import plot_topic_trends                  # noqa: E402


def latest(results_dir: Path, pattern: str, filename: str):
    hits = sorted(p for p in results_dir.glob(f"{pattern}/{filename}")
                  if "smoke" not in p.parent.name)
    return hits[-1] if hits else None


def short_label(words, n: int = 3) -> str:
    return ", ".join(words[:n])


def main() -> int:
    p = argparse.ArgumentParser(description="Temporal topic analysis")
    p.add_argument("--config", default=None)
    p.add_argument("--min-docs", type=int, default=30,
                   help="Windows below this are excluded from trend fitting")
    args = p.parse_args()

    cfg = load_config(args.config)
    set_seed(cfg["seed"])
    tcfg = cfg["temporal"]
    run = new_run(cfg, tag="temporal")
    print(f"Run: {run.run_id}\n")

    windows = [f"{a}-{b}" for a, b in cfg["corpus"]["time_windows"]]
    rows = [json.loads(l) for l in open(cfg.path("tokens"), encoding="utf-8") if l.strip()]
    doc_windows = [r["time_window"] for r in rows]
    print(f"Documents: {len(rows)}  Windows: {windows}\n")

    results_dir = cfg.path("results_dir")
    import numpy as np

    outputs = {}
    for label, pattern, jsonname in (
        ("GA-Optimized LDA", "*_ga_lda", "ga_lda.json"),
        ("BERTopic", "*_bertopic", "bertopic.json"),
    ):
        payload_path = latest(results_dir, pattern, jsonname)
        if payload_path is None:
            print(f"  MISSING: {label}")
            continue
        dt_path = payload_path.parent / "doc_topics.npy"
        if not dt_path.exists():
            print(f"  MISSING doc_topics.npy for {label}")
            continue

        with open(payload_path, encoding="utf-8") as fh:
            payload = json.load(fh)
        doc_topics = np.load(dt_path)
        if doc_topics.shape[0] != len(rows):
            print(f"  SKIP {label}: {doc_topics.shape[0]} rows vs {len(rows)} documents")
            continue

        props, counts = window_proportions(
            doc_topics, windows, doc_windows,
            normalize_within_window=tcfg.get("normalize_within_window", True),
        )
        trends, excluded = topic_trends(props, windows, counts, min_docs=args.min_docs)
        labels = [short_label(t) for t in payload["topics"]]

        print("=" * 78)
        print(f"{label}  ({payload['run_id']})")
        print("=" * 78)
        print("Documents per window: " +
              ", ".join(f"{w}: {c}" for w, c in zip(windows, counts)))
        if excluded:
            print(f"  Excluded from trend fitting (< {args.min_docs} docs): {excluded}")
            print("  Their shares are still reported, but a slope through them is not meaningful.")

        print(f"\n{'Topic':<40}" + "".join(f"{w:>12}" for w in windows) + f"{'trend':>12}")
        print("-" * (40 + 12 * len(windows) + 12))
        for tr in trends:
            name = f"{tr['topic']:>2}: {labels[tr['topic']]}"[:38]
            cells = "".join(
                f"{(tr['by_window'][w] if tr['by_window'][w] is not None else float('nan')):>12.3f}"
                for w in windows
            )
            print(f"{name:<40}{cells}{tr['trend']:>12}")

        rising = [t for t in trends if t["trend"] == "rising"][:3]
        declining = [t for t in trends if t["trend"] == "declining"][:3]
        print(f"\n  Rising:    " + "; ".join(
            f"{labels[t['topic']]} ({t['absolute_change']:+.3f})" for t in rising) or "  none")
        print(f"  Declining: " + "; ".join(
            f"{labels[t['topic']]} ({t['absolute_change']:+.3f})" for t in declining) or "  none")

        fig_name = "fig_trends_ga_lda.png" if "GA" in label else "fig_trends_bertopic.png"
        fig = plot_topic_trends(
            props, windows, counts, labels, run.file(fig_name),
            top_n=min(8, props.shape[1]),
            title=f"Thematic evolution ({label})",
        )
        if "GA" in label:
            import shutil
            figs = cfg.path("figures_dir")
            figs.mkdir(parents=True, exist_ok=True)
            shutil.copy(fig, figs / fig_name)

        outputs[label] = {
            "run_id": payload["run_id"],
            "windows": windows,
            "counts": counts,
            "excluded_from_trend_fit": excluded,
            "labels": labels,
            "proportions": [[None if v != v else round(float(v), 6) for v in row]
                            for row in props],
            "trends": trends,
        }
        print()

    payload = {
        "experiment": "EXP-005",
        "run_id": run.run_id,
        "primary_model": "GA-Optimized LDA",
        "primary_model_rationale": (
            "LDA assigns a topic distribution to 100% of documents; BERTopic leaves "
            "17.1% unclustered, so window proportions from it omit part of the corpus."
        ),
        "normalize_within_window": tcfg.get("normalize_within_window", True),
        "min_docs_for_trend_fit": args.min_docs,
        "models": outputs,
    }
    run.write_json("temporal.json", payload)

    root = run.path.parents[2]
    print(f"Run artefacts -> {run.path.relative_to(root)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
