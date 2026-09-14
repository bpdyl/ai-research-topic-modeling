"""Approximate prose word count, the way a marker would count it.

    python wordcount.py

The brief caps the coursework at 4,500 words, so it is worth knowing where the
document actually stands. Counting `wc -w` on a .tex file overstates badly:
every \\textbf, every table cell and every figure caption is counted as prose.

This strips the things a marker would not count as body text -- comments,
preamble, tabular bodies, figure/table environments including captions, maths,
bibliography -- and reports what is left. It is an estimate, not an authority:
different markers count captions and headings differently, so treat the number
as a range finder rather than a verdict.
"""

from __future__ import annotations

import re
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent


def prose_words(path: Path, drop_preamble: bool = False) -> int:
    text = path.read_text(encoding="utf-8")

    text = re.sub(r"(?<!\\)%.*", "", text)                     # comments
    if drop_preamble:
        text = text.split(r"\maketitle", 1)[-1]
    # float environments carry captions and tabular data, not body prose
    for env in ("table", "table*", "figure", "figure*", "tabular", "center",
                "thebibliography", "equation", "align", "IEEEkeywords"):
        text = re.sub(rf"\\begin\{{{re.escape(env)}\}}.*?\\end\{{{re.escape(env)}\}}",
                      " ", text, flags=re.S)
    text = re.sub(r"\$[^$]*\$", " x ", text)                   # inline maths
    text = re.sub(r"\\[a-zA-Z@]+\*?(\[[^\]]*\])?", " ", text)  # commands
    text = re.sub(r"[{}~&\\]", " ", text)                      # leftovers
    words = [w for w in text.split() if re.search(r"[A-Za-z0-9]", w)]
    return len(words)


def main() -> None:
    rows = [
        ("Task 1  (main.tex body, excl. Task 2)", None),
        ("Task 2  (task2.tex)", HERE / "task2.tex"),
        ("Task 2  (appendix_task2.tex)", HERE / "appendix_task2.tex"),
    ]

    main_tex = (HERE / "main.tex").read_text(encoding="utf-8")
    marker = "P A P E R   2"
    if marker not in main_tex:
        raise SystemExit(f"cannot find the paper-1/paper-2 split marker "
                         f"({marker!r}) in main.tex")
    task1_only = main_tex.split(marker)[0]

    tmp = HERE / ".task1_only.tmp.tex"
    tmp.write_text(task1_only, encoding="utf-8")
    t1 = prose_words(tmp, drop_preamble=True)

    # Markers differ on whether the abstract counts towards the body limit,
    # so report both rather than silently pick one.
    no_abstract = re.sub(r"\\begin\{abstract\}.*?\\end\{abstract\}", " ",
                         task1_only, flags=re.S)
    tmp.write_text(no_abstract, encoding="utf-8")
    t1_no_abs = prose_words(tmp, drop_preamble=True)
    tmp.unlink()

    t2 = prose_words(HERE / "task2.tex")
    t2a = prose_words(HERE / "appendix_task2.tex")

    print(f"{'Task 1 body (incl. abstract)':<34}{t1:>7}")
    print(f"{'Task 1 body (excl. abstract)':<34}{t1_no_abs:>7}")
    print(f"{'Task 2 body':<34}{t2:>7}")
    print(f"{'Task 2 appendix':<34}{t2a:>7}")
    print("-" * 41)
    print(f"{'Task 2 total':<34}{t2 + t2a:>7}")
    print(f"{'COMBINED':<34}{t1 + t2 + t2a:>7}")
    print()
    print("How the brief's limits actually apply")
    print("-------------------------------------")
    print("The '6 pages of A4, up to 4000 words, single-column' spec sits under")
    print("TASK 1's deliverables, not the coursework as a whole. Task 2's brief")
    print("states mark allocations and required content but NO page or word")
    print("limit. The admin block's '4500 words (total)' is best read as the")
    print("1-page proposal plus Task 1's 4,000-word paper.")
    print()
    print(f"  Task 1 vs its 4,000-word cap, incl. abstract : {t1:,}"
          + ("  OVER" if t1 > 4000 else "  ok"))
    print(f"  Task 1 vs its 4,000-word cap, excl. abstract : {t1_no_abs:,}"
          + ("  OVER" if t1_no_abs > 4000 else "  ok"))
    print(f"  Task 2 against a stated cap           : none stated")
    print()
    if t1_no_abs > 4000:
        print(f"Task 1 is ~{t1_no_abs - 4000:,} words over even excluding its")
        print("abstract. Trim before submitting.")
    elif t1 > 4000:
        print("Task 1 is over the cap only if the abstract is counted towards")
        print("it. Most markers count the body; check the convention if in doubt.")


if __name__ == "__main__":
    main()
