"""
Step 08 — LLM-assisted topic labelling (deliverable #3).

Builds the labelling pack (top words + representative document titles + the
exact prompt used) for every model, merges in the drafted labels, and writes a
review sheet for the team.

The `human_reviewed` flag starts False for every topic and is never set by this
script. The proposal claims labels are "reviewed and finalized by the team";
only a person editing the review sheet can make that true.

Usage:
    python scripts/08_label_topics.py
"""

from __future__ import annotations

import argparse
import csv
import glob
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from nrtm.config import load_config, new_run, set_seed                 # noqa: E402
from nrtm.labeling.llm_labels import build_labelling_pack, merge_labels, PROMPT_TEMPLATE  # noqa: E402

SOURCES = [
    ("GA-Optimized LDA", "results/runs/*_ga_lda/ga_lda.json"),
    ("BERTopic", "results/runs/*_bertopic/bertopic.json"),
    ("Standard LDA", "results/runs/*_lda_baseline/lda_baseline.json"),
]


def main() -> int:
    p = argparse.ArgumentParser(description="LLM-assisted topic labelling")
    p.add_argument("--config", default=None)
    p.add_argument("--labels", default="data/processed/llm_labels_draft.json")
    args = p.parse_args()

    cfg = load_config(args.config)
    set_seed(cfg["seed"])
    run = new_run(cfg, tag="label_topics")
    print(f"Run: {run.run_id}\n")

    import numpy as np

    frozen = [json.loads(l) for l in
              open(cfg.path("frozen_corpus"), encoding="utf-8") if l.strip()]
    titles = [r["title"] for r in frozen]

    labels_path = Path(args.labels)
    if not labels_path.is_absolute():
        labels_path = cfg.path("frozen_corpus").parents[2] / labels_path
    drafted = json.loads(labels_path.read_text(encoding="utf-8"))
    provenance = drafted.get("_provenance", {})
    print(f"Label source: {labels_path.name}")
    print(f"  drafted by : {provenance.get('drafted_by', 'unknown')}")
    print(f"  status     : {provenance.get('status', 'unknown')}\n")

    all_out = {}
    review_rows = []

    for model_name, pattern in SOURCES:
        hits = sorted(h for h in glob.glob(pattern) if "smoke" not in h)
        if not hits:
            print(f"  MISSING: {model_name}")
            continue
        payload = json.loads(Path(hits[-1]).read_text(encoding="utf-8"))
        dt_path = Path(hits[-1]).parent / "doc_topics.npy"
        doc_topics = np.load(dt_path) if dt_path.exists() else None
        if doc_topics is None or doc_topics.shape[0] != len(titles):
            doc_topics = np.zeros((len(titles), len(payload["topics"])))

        pack = build_labelling_pack(payload["topics"], doc_topics, titles, n_representative=4)
        merged = merge_labels(pack, drafted.get(model_name, {}))
        all_out[model_name] = {"run_id": payload["run_id"], "topics": merged}

        n_labelled = sum(1 for e in merged if e["llm_label"])
        print(f"{model_name:<20} {len(merged)} topics, {n_labelled} drafted labels")
        for e in merged:
            print(f"   {e['topic_id']:>2}: {e['llm_label'] or '(none)'}")
            review_rows.append({
                "model": model_name,
                "topic_id": e["topic_id"],
                "top_words": ", ".join(e["top_words"]),
                "llm_draft_label": e["llm_label"] or "",
                "final_label_EDIT_ME": "",
                "reviewer_initials": "",
                "accept_reject_revise": "",
            })
        print()

    review_path = run.file("topic_label_review_sheet.csv")
    with open(review_path, "w", encoding="utf-8", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(review_rows[0]))
        w.writeheader()
        w.writerows(review_rows)

    payload = {
        "experiment": "EXP-006",
        "run_id": run.run_id,
        "prompt_template": PROMPT_TEMPLATE,
        "label_provenance": provenance,
        "models": all_out,
        "human_review_status": "PENDING — no topic has human_reviewed=true",
        "topics_awaiting_review": len(review_rows),
    }
    run.write_json("topic_labels.json", payload)

    root = run.path.parents[2]
    print(f"Review sheet  -> {review_path.relative_to(root)}")
    print(f"Run artefacts -> {run.path.relative_to(root)}")
    print("\nHUMAN REVIEW PENDING for all "
          f"{len(review_rows)} topics. The proposal's claim that labels are "
          "'reviewed and finalized by the team' is not yet met.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
