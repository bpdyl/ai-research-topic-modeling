"""Pre-flight checks for the LaTeX report, for when you cannot compile locally.

    python check_report.py

There is no TeX toolchain on the authoring machine (the build happens on
Overleaf), so the usual way of catching a broken reference -- compile and read
the log -- is not available. These checks stand in for the parts of that log
that matter most, and all of them are things Overleaf would only tell you after
an upload round-trip:

  1. Citation order.   The brief asks for references numbered from 1 in the
     order they are first cited. With a manual `thebibliography`, numbering
     follows \\bibitem order, so the two lists have to agree by construction.
  2. Undefined / unused keys, in both directions.
  3. Undefined \\ref targets and duplicate \\label definitions.
  4. Every \\includegraphics resolves to a file that exists.
  5. Figure count, against the report's own target.

Exits non-zero if anything fails, so it can gate a commit.
"""

from __future__ import annotations

import re
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent

# Minimum image count per document, as agreed for this submission: the combined
# report carries both papers, the standalone carries Paper 2 only.
FIG_TARGETS = {"main.tex": 23, "task2_standalone.tex": 16}
FIG_TARGET_DEFAULT = 16


def target_doc() -> str:
    """Which document to check: `--doc <file>`, default main.tex."""
    if "--doc" in sys.argv:
        return sys.argv[sys.argv.index("--doc") + 1]
    return "main.tex"


def check_environments(doc: str, failures: list) -> None:
    """\\begin/\\end balance, the error that costs an Overleaf round-trip.

    A mismatched environment is the single most common way this document could
    fail to compile, and it is the one a text editor will not catch. Checked as
    a stack so the report names the environment and the line, not just a count.
    """
    stack, line_no = [], 0
    for line_no, line in enumerate(doc.splitlines(), 1):
        for m in re.finditer(r"\\(begin|end)\{([^}]+)\}", line):
            kind, env = m.group(1), m.group(2)
            if kind == "begin":
                stack.append((env, line_no))
            else:
                if not stack:
                    failures.append(
                        rf"line {line_no}: \end{{{env}}} with nothing open")
                elif stack[-1][0] != env:
                    open_env, open_line = stack[-1]
                    failures.append(
                        rf"line {line_no}: \end{{{env}}} closes "
                        rf"\begin{{{open_env}}} from line {open_line}")
                    stack.pop()
                else:
                    stack.pop()
    for env, ln in stack:
        failures.append(rf"\begin{{{env}}} at line {ln} is never closed")
    print("environments: balanced" if not failures else "environments: PROBLEM")


def expand(path: Path, depth: int = 0) -> str:
    """Inline \\input{} so the checks see the document as LaTeX will."""
    if depth > 10:
        raise RuntimeError("\\input nesting too deep")
    text = path.read_text(encoding="utf-8")
    out = []
    for line in text.splitlines():
        m = re.match(r"\s*\\input\{([^}]+)\}\s*$", line)
        if m:
            child = HERE / (m.group(1) if m.group(1).endswith(".tex")
                            else m.group(1) + ".tex")
            if not child.exists():
                out.append(f"%% MISSING INPUT: {child.name}")
                continue
            out.append(expand(child, depth + 1))
        else:
            out.append(line)
    return "\n".join(out)


def strip_comments(text: str) -> str:
    """Remove LaTeX comments without eating an escaped percent sign."""
    return re.sub(r"(?<!\\)%.*", "", text)


def check_escapes(path: Path, failures: list) -> None:
    """Unescaped `%` in body text --- the one LaTeX error that stays silent.

    An unescaped `%` comments out the rest of its line, so the text is simply
    absent from the PDF and no warning is raised anywhere. That makes it worth
    a static check even though everything else here would be caught by reading
    a compile log.

    Deliberately narrow. An unescaped `_` was checked here too and produced
    almost nothing but false positives, because `_` is legal inside maths and
    inside `\\texttt`, and inline maths spans line breaks freely. It also does
    not need a static check: it raises a loud "Missing $ inserted" that names
    the line. Checking one thing accurately beats checking two things noisily.
    """
    MATH_ENVS = ("equation", "equation*", "align", "align*", "eqnarray",
                 "eqnarray*", "gather", "gather*", "displaymath", "array")
    text = path.read_text(encoding="utf-8")
    in_math = False

    for n, line in enumerate(text.splitlines(), 1):
        if re.search(r"\\begin\{(" + "|".join(re.escape(e) for e in MATH_ENVS)
                     + r")\}", line):
            in_math = True
        stripped_line = line
        if re.search(r"\\end\{(" + "|".join(re.escape(e) for e in MATH_ENVS)
                     + r")\}", line):
            in_math = False
            continue
        if in_math:
            continue

        if line.lstrip().startswith("%"):
            continue

        probe = re.sub(r"\\%", "", stripped_line)            # already escaped

        # A trailing % is the standard line-continuation idiom, not an error.
        body = probe.rstrip()
        if body.endswith("%"):
            body = body[:-1]
        m = re.search(r"(?<!\\)%", body)
        if m and body[:m.start()].strip():
            failures.append(
                f"{path.name}:{n}: unescaped '%' mid-line -> "
                f"{line.strip()[:60]}")


def main() -> int:
    root = target_doc()
    print(f"document  : {root}\n")
    doc_raw = expand(HERE / root)
    doc = strip_comments(doc_raw)

    failures = []

    # This document is two self-contained papers, each with its own reference
    # list numbered from 1. Each bibliography is therefore checked against the
    # citations that precede it, not against the document as a whole.
    bibs = list(re.finditer(
        r"\\begin\{thebibliography\}(.*?)\\end\{thebibliography\}", doc, re.S))
    if not bibs:
        failures.append("no thebibliography found")
        bibs = []

    print(f"papers    : {len(bibs)}, each with its own reference list\n")

    # Split the document into papers, NOT at the bibliographies. Each paper's
    # reference list sits before its appendix, and the appendix cites the
    # repository entry, so a citation can legitimately appear *after* the
    # bibliography that defines it. Slicing at bibliography boundaries would
    # misreport that as "listed but never cited".
    # Each paper begins at its \paperhead call. That marker is real markup, so
    # it survives comment stripping; the "P A P E R n" banners do not.
    marks = [m.start() for m in re.finditer(r"\\paperhead\b", doc)
             if "newcommand" not in doc[max(0, m.start() - 60):m.start()]]
    bounds = ([0] + marks[1:] if marks else [0]) + [len(doc)]

    for n, bib in enumerate(bibs, 1):
        lo = max((b for b in bounds if b <= bib.start()), default=0)
        hi = min((b for b in bounds if b > bib.start()), default=len(doc))
        section = doc[lo:hi]

        cite_order, seen = [], set()
        for m in re.finditer(r"\\cite\{([^}]*)\}", section):
            for key in (k.strip() for k in m.group(1).split(",")):
                if key and key not in seen:
                    seen.add(key)
                    cite_order.append(key)

        bib_order = re.findall(r"\\bibitem\{([^}]+)\}", bib.group(1))

        print(f"  Paper {n}: {len(cite_order)} distinct citations, "
              f"{len(bib_order)} bibitems")

        undefined = [k for k in cite_order if k not in bib_order]
        unused = [k for k in bib_order if k not in cite_order]
        if undefined:
            failures.append(f"paper {n}: cited but not listed: {undefined}")
        if unused:
            failures.append(f"paper {n}: listed but never cited: {unused}")

        if "--order" in sys.argv or undefined:
            print(f"    required \\bibitem order for paper {n}:")
            for i, key in enumerate(cite_order, 1):
                mark = "  MISSING" if key in undefined else ""
                print(f"      [{i:2d}] {key}{mark}")

        if not undefined and not unused:
            mismatch = [i for i, (c, b)
                        in enumerate(zip(cite_order, bib_order), 1) if c != b]
            if mismatch:
                failures.append(
                    f"paper {n}: {len(mismatch)} entries out of "
                    f"first-citation order")
                print(f"    NUMBERING NOT SEQUENTIAL. Expected order:")
                for i, key in enumerate(cite_order, 1):
                    flag = ("  <-- currently " + str(bib_order.index(key) + 1)
                            if bib_order[i - 1] != key else "")
                    print(f"      [{i:2d}] {key}{flag}")
            else:
                print(f"    order matches: [1]..[{len(bib_order)}] "
                      f"sequential  OK")

    # keys must be globally unique: \bibitem defines them document-wide
    all_keys = [k for b in bibs
                for k in re.findall(r"\\bibitem\{([^}]+)\}", b.group(1))]
    dupe_keys = sorted({k for k in all_keys if all_keys.count(k) > 1})
    if dupe_keys:
        failures.append(
            f"same key in more than one bibliography (numbering will be "
            f"wrong): {dupe_keys}")
    print()

    # --------------------------------------------------------- labels/refs
    labels = re.findall(r"\\label\{([^}]+)\}", doc)
    dupes = {l for l in labels if labels.count(l) > 1}
    if dupes:
        failures.append(f"duplicate labels: {sorted(dupes)}")

    refs = set(re.findall(r"\\(?:eq)?ref\{([^}]+)\}", doc))
    dangling = sorted(refs - set(labels))
    if dangling:
        failures.append(f"\\ref to undefined labels: {dangling}")
    print(f"\nlabels    : {len(labels)} defined, {len(refs)} referenced"
          + ("  OK" if not dupes and not dangling else ""))

    # -------------------------------------------------------------- figures
    graphics = re.findall(r"\\includegraphics(?:\[[^\]]*\])?\{([^}]+)\}", doc)
    matlab = re.findall(r"\\matlabfig\{([^}]+)\}", doc)
    # `#1` and friends come from the \matlabfig definition in the preamble,
    # not from a real figure inclusion.
    graphics = [g for g in graphics if "#" not in g]
    missing, pending = [], []
    for g in graphics:
        if not (HERE / "figures" / g).exists():
            missing.append(g)
    for m in matlab:
        if not (HERE / "figures" / m).exists():
            pending.append(m)
    if missing:
        failures.append(f"\\includegraphics missing files: {missing}")

    total = len(graphics) + len(matlab)
    fig_target = FIG_TARGETS.get(root, FIG_TARGET_DEFAULT)
    print(f"figures   : {total} images "
          f"({len(graphics)} direct + {len(matlab)} MATLAB placeholders)")
    if total < fig_target:
        failures.append(f"only {total} images, target is {fig_target}")
    else:
        print(f"            target of {fig_target} met  OK")
    if pending:
        print(f"            {len(pending)} MATLAB screenshot(s) not yet "
              f"supplied (placeholder will render):")
        for p in pending:
            print(f"              - {p}")

    # --------------------------------------------------------- environments
    print()
    env_failures = []
    check_environments(doc, env_failures)
    failures.extend(env_failures)

    esc_failures = []
    for name in (root, "task2.tex", "appendix_task2.tex"):
        p = HERE / name
        if p.exists():
            check_escapes(p, esc_failures)
    print(f"escapes   : {'clean' if not esc_failures else 'PROBLEM'}"
          f" ({len(esc_failures)} suspect)")
    failures.extend(esc_failures[:15])

    # ---------------------------------------------------------------- verdict
    print()
    if failures:
        print("FAILED:")
        for f in failures:
            print(f"  - {f}")
        return 1
    print("all checks passed")
    return 0


if __name__ == "__main__":
    sys.exit(main())
