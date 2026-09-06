# -*- coding: utf-8 -*-
"""
Step 12 — Interpretability panel (EXP-009).

Aggregates three independent blind ratings of all 39 topics plus an independent
label review, and computes inter-rater agreement.

The raters are **three independent LLM judges**, not the three human raters the
proposal specifies. Each ran in a separate context and saw only the shuffled
top-10 word lists and the rubric: no model names, no draft labels, and no
sight of another rater's output. Agreement between them is therefore real and
computable — which is what makes the proposal's promised agreement statistic
reportable at all — but they are automated raters and the paper says so.

Raw ratings live in data/processed/ratings/ so the aggregation is auditable and
re-runnable without re-rating.

Usage:
    python scripts/12_interpretability_panel.py
"""
import io, json, os, statistics, sys

REPO = r"d:\Softwarica\Advanced ML\Main Assignment\ai-research-topic-modeling"
SCR = os.path.join(REPO, "data", "processed", "ratings")
os.chdir(REPO)
sys.path.insert(0, "src")

from nrtm.evaluation.agreement import (
    krippendorff_alpha_ordinal, percent_agreement, interpret_alpha)

pack = json.load(io.open("data/processed/blind_rating_pack.json", encoding="utf-8"))
raters = {k: json.load(io.open(os.path.join(SCR, f"rater_{k}.json"), encoding="utf-8"))
          for k in ("A", "B", "C")}
review = json.load(io.open(os.path.join(SCR, "label_review.json"), encoding="utf-8"))

items = [str(r["item"]) for r in pack]
matrix = [[raters[k][i]["rating"] for i in items] for k in ("A", "B", "C")]

alpha = krippendorff_alpha_ordinal(matrix)
print("=" * 68)
print("INTER-RATER AGREEMENT — 3 independent raters, 39 topics")
print("=" * 68)
print(f"  Krippendorff's alpha (ordinal) : {alpha:.4f}  -> {interpret_alpha(alpha)}")
print(f"  exact agreement                : {percent_agreement(matrix, 0):.1%}")
print(f"  agreement within 1 point       : {percent_agreement(matrix, 1):.1%}")

# Per-model means from the panel
by_model = {}
for r in pack:
    scores = [raters[k][str(r["item"])]["rating"] for k in ("A", "B", "C")]
    by_model.setdefault(r["model"], []).append(statistics.mean(scores))

print("\n" + "=" * 68)
print("INTERPRETABILITY — panel mean of 3 raters")
print("=" * 68)
print(f"  {'Model':<22}{'n':>4}{'mean':>8}{'sd':>7}{'>=4':>7}")
print("  " + "-" * 46)
summary = {}
for m, vals in sorted(by_model.items(), key=lambda kv: -statistics.mean(kv[1])):
    mean = statistics.mean(vals)
    sd = statistics.stdev(vals) if len(vals) > 1 else 0.0
    ge4 = sum(1 for v in vals if v >= 4)
    summary[m] = {"n_topics": len(vals), "mean": round(mean, 3), "sd": round(sd, 3),
                  "n_ge_4": ge4, "pct_ge_4": round(100 * ge4 / len(vals), 1),
                  "per_topic_means": [round(v, 3) for v in vals]}
    print(f"  {m:<22}{len(vals):>4}{mean:>8.2f}{sd:>7.2f}{ge4:>7}")

# Largest disagreements — worth naming in the paper
print("\n  largest rater disagreements:")
spread = sorted(pack, key=lambda r: -(max(raters[k][str(r['item'])]['rating'] for k in 'ABC')
                                      - min(raters[k][str(r['item'])]['rating'] for k in 'ABC')))
for r in spread[:4]:
    s = [raters[k][str(r["item"])]["rating"] for k in "ABC"]
    if max(s) - min(s) == 0:
        break
    print(f"    item {r['item']:>2} ({r['model'][:18]:<18}) A={s[0]} B={s[1]} C={s[2]}  {r['words'][:52]}...")

# Label review outcome
print("\n" + "=" * 68)
print("LABEL REVIEW — independent reviewer, 39 labels")
print("=" * 68)
v = {}
for i, e in review.items():
    v[e["verdict"]] = v.get(e["verdict"], 0) + 1
for k in ("accept", "revise", "reject"):
    print(f"  {k:<10}{v.get(k,0):>3}  ({100*v.get(k,0)/39:.0f}%)")

by_model_v = {}
for r in pack:
    e = review[str(r["item"])]
    by_model_v.setdefault(r["model"], {}).setdefault(e["verdict"], 0)
    by_model_v[r["model"]][e["verdict"]] += 1
print("\n  by model:")
for m, d in by_model_v.items():
    tot = sum(d.values())
    print(f"    {m:<22} accept {d.get('accept',0)}/{tot}, revise {d.get('revise',0)}, reject {d.get('reject',0)}")

# ── Persist ───────────────────────────────────────────────────────────
out = {
    "_provenance": {
        "panel": "THREE INDEPENDENT LLM RATERS (claude-opus-5), separate contexts, blind to model identity",
        "NOT_HUMAN": "These are NOT the three human raters the proposal specifies. They are three "
                     "independent automated raters. Inter-rater agreement between them is real and "
                     "computable, but they are LLM judges and the paper says so.",
        "independence": "Each rater received only the shuffled top-10 word lists and the rating "
                        "rubric. No rater saw another's output, the model names, or the draft labels.",
        "rated_on": "2026-09-06",
        "n_items": 39,
    },
    "agreement": {
        "krippendorff_alpha_ordinal": round(alpha, 4),
        "interpretation": interpret_alpha(alpha),
        "exact_agreement": round(percent_agreement(matrix, 0), 4),
        "within_one_point": round(percent_agreement(matrix, 1), 4),
    },
    "summary": summary,
    "per_item": [
        {"item": r["item"], "model": r["model"], "topic_id": r["topic_id"],
         "words": r["words"],
         "ratings": {k: raters[k][str(r["item"])]["rating"] for k in "ABC"},
         "mean": round(statistics.mean([raters[k][str(r["item"])]["rating"] for k in "ABC"]), 3),
         "label_verdict": review[str(r["item"])]["verdict"],
         "final_label": review[str(r["item"])]["final_label"],
         "review_note": review[str(r["item"])]["note"]}
        for r in pack
    ],
    "label_review": {"accept": v.get("accept", 0), "revise": v.get("revise", 0),
                     "reject": v.get("reject", 0), "by_model": by_model_v},
}
io.open("data/processed/interpretability_panel.json", "w", encoding="utf-8").write(
    json.dumps(out, indent=2, ensure_ascii=False))
print("\nwritten: data/processed/interpretability_panel.json")

# Final labels, keyed by model/topic
final = {}
for r in pack:
    final.setdefault(r["model"], {})[str(r["topic_id"])] = review[str(r["item"])]["final_label"]
io.open("data/processed/final_topic_labels.json", "w", encoding="utf-8").write(
    json.dumps({"_provenance": {
        "drafted_by": "claude-opus-5 (EXP-006)",
        "reviewed_by": "independent LLM reviewer, separate context, 2026-09-06",
        "outcome": f"{v.get('accept',0)} accepted, {v.get('revise',0)} revised, {v.get('reject',0)} rejected",
        "note": "Rejected labels were replaced with honest low-coherence descriptions rather than "
                "plausible-sounding invented themes.",
    }, **final}, indent=2, ensure_ascii=False))
print("written: data/processed/final_topic_labels.json")
