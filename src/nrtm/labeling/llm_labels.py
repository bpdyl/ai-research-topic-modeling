"""
LLM-assisted topic labelling.

The proposal commits to labels "drafted by an LLM, reviewed and finalized by the
team". Two things follow from that, and both are handled here:

1. **It is a method step, not a writing shortcut.** The assignment brief's
   integrity clause concerns tools that write the submission. Using a language
   model as an *instrument* — mapping a top-word list to a human-readable name,
   under human review — is a documented methodological choice, and it is
   recorded as one: exact prompt template, model name, and the before/after of
   human review are all persisted (OQ-007).

2. **It must be reproducible.** A label nobody can regenerate is not evidence.
   The prompt is built deterministically from the topic's top words and its
   most representative documents, so the same inputs produce the same request.

Representative documents matter: `image, system, feature, detection, accuracy`
is ambiguous from words alone, but the titles of the documents that load most
heavily on it usually are not.
"""

from __future__ import annotations

from typing import Sequence

PROMPT_TEMPLATE = """You are labelling topics from a topic model fitted on titles and abstracts of
AI/ML research papers with at least one Nepal-affiliated author, published 2015-2025.

For the topic below, give a short human-readable label of 2-5 words naming the
research theme. Use domain terminology, not generic words like "data analysis".
If the top words look incoherent, say so rather than inventing a theme.

Topic {topic_id}
Top words: {top_words}
Most representative paper titles:
{representative_titles}

Reply with only the label."""


def representative_documents(
    doc_topics,
    topic_id: int,
    titles: Sequence[str],
    n: int = 5,
) -> list[str]:
    """Titles of the documents loading most heavily on a topic."""
    import numpy as np

    column = np.asarray(doc_topics, dtype=float)[:, topic_id]
    order = np.argsort(-column)[:n]
    return [titles[i] for i in order if column[i] > 0]


def build_prompt(topic_id: int, top_words: Sequence[str], titles: Sequence[str]) -> str:
    return PROMPT_TEMPLATE.format(
        topic_id=topic_id,
        top_words=", ".join(top_words),
        representative_titles="\n".join(f"  - {t}" for t in titles) or "  (none)",
    )


def build_labelling_pack(
    topics: Sequence[Sequence[str]],
    doc_topics,
    titles: Sequence[str],
    n_representative: int = 5,
) -> list[dict]:
    """One entry per topic: words, representative titles, and the exact prompt."""
    pack = []
    for i, words in enumerate(topics):
        reps = representative_documents(doc_topics, i, titles, n=n_representative)
        pack.append({
            "topic_id": i,
            "top_words": list(words),
            "representative_titles": reps,
            "prompt": build_prompt(i, words, reps),
        })
    return pack


def merge_labels(pack: Sequence[dict], labels: dict) -> list[dict]:
    """Attach drafted labels to the pack, leaving human review explicit.

    `human_reviewed` starts False for every topic. It is not set here and must
    not be set by any script: only a person can flip it, and the paper's claim
    that labels were "reviewed and finalised by the team" depends on that
    distinction being real.
    """
    out = []
    for entry in pack:
        e = dict(entry)
        e["llm_label"] = labels.get(str(entry["topic_id"]), labels.get(entry["topic_id"]))
        e["final_label"] = None
        e["human_reviewed"] = False
        out.append(e)
    return out
