# Validated report revision - 15 September 2026

## Follow-up: automated labelling app and Task 2 wording

The proposal's automated labelling step was already represented by assistant-session
drafts for all 39 topics. A new Streamlit app (`apps/topic_labeler.py`) adds native
OpenAI/Claude/Gemini API adapters, evidence preview, resumable generation, provenance,
explicit human review and exports. See `apps/README.md`. All 126 tests pass, including
15 new workflow/interface tests. Live provider calls remain untested because no
API key was configured. The paper retains its archived labeling experiment and
does not claim new API-generated results. Its updated word estimate is 4,220 prose
or 4,500 including captions/headings under the supplied counter.

Task 2's incorrect Toolbox claims have been corrected to base MATLAB. Direct
re-reading of the brief confirms it names MATLAB generally before giving Toolbox
links; the earlier requirements summary was too restrictive. All three required
parts are covered, with runtime and packaging limits in `TASK2_COMPLETENESS.md`.
These updates supersede conflicting historical descriptions below.

## Earlier revision

This revision implements the review in `validation_review_2026-09-14.md`.
That review describes the earlier draft; the status below supersedes its open
presentation and benchmark findings.

## Approved presentation

The team reports professor approval for Task 1 up to 4,500 words, no Task 2 word
limit, the existing two-column format, and extra pages. Both reports describe
equal collaborative contributions by Bibek Paudyal, Sajan Mahat and Siddhartha
Bhatta, following the team's instruction. This does not change the provenance
of the automated topic assessments.

## What changed

- Task 1 now has a fuller introduction, a separate Related Work section, explicit
  research questions, ten figures, and a clearer account of departures from the
  proposal. The revised text is approximately 4,084 prose words, or 4,364 including
  captions/headings under the supplied counter; references, table cells and
  equations are excluded from these estimates.
- Metric definitions now distinguish the search objective's top-25 diversity
  from the archived comparison's top-10 diversity. In-sample perplexity is no
  longer presented as held-out evidence. The random-search control and automated
  judgments are reported without claiming a universal GA advantage or a complete
  inversion of model rankings.
- Ten unused initialisation seeds per fixed LDA configuration were run (30 fits).
  Matched-word stability was 0.2718 for standard LDA, 0.3525 for GA-LDA and 0.3581
  for random-search LDA. These are same-corpus fixed-configuration checks, not
  repetitions of the searches or a held-out generalisation experiment.
- Temporal figures use audited labels and disclose the eight-record first window.
  BERTopic's assigned-document denominators are 5/8, 72/93, 249/299 and 564/674.
  A 2,000-resample conditional bootstrap and an all-document denominator check
  support the later-window education-related increase, within the stated limits.
- A residual retrieval false positive was identified: OpenAlex W2291375252,
  a 2016 paper about students' learning approaches rather than AI. Removing it
  from reporting alone reduces the earliest GA education share from about 10.8%
  to 2.1%. The frozen corpus and historical fits were preserved; the paper explains
  why this is a sensitivity analysis and why corpus adjudication remains needed.
- Task 2 now uses official CEC2005 F6/F9 shift vectors. All 180 trials were rerun;
  the tables, convergence distributions and discussion were regenerated. Two-sided
  Mann-Whitney comparisons now use Holm correction across 12 comparisons. The
  earlier local-shift results are preserved in an explicitly named archive.
- Added the official-data MATLAB benchmark implementation and corrected optimiser
  settings in the report. The new MATLAB listing was not executed in this revision.
- Shared LaTeX sources now build separate Task 1 and Task 2 submission PDFs, as well
  as the combined working report. Figure extents, equation/table overflows and
  figure placement were corrected after rendering.

## Verification and remaining limits

All 111 Python tests pass, including controller agreement with scikit-fuzzy,
official benchmark values/budget enforcement and empty temporal-window handling.
The original local-instance optimiser regression remains explicitly pinned to
its archived shift; it is not a claim that every optimiser must beat random
sampling on every official instance. scikit-fuzzy emits upstream NumPy
deprecation warnings without test failures.

Both standalone PDFs compile and the source checker verifies citations, labels,
figure references and balanced environments. Rendered pages were inspected.
The full notebook and full historical topic searches were not rerun in this
revision; archived outputs were checked and additional validation was run
separately. Task 2's earlier controller-tuning trials were retained, not rerun.

Human topic validation is still outstanding: saved assessments came from LLM
judges in separate contexts. The supplied proposal itself does not explicitly
promise three human raters; older project notes saying otherwise are inaccurate.
The corpus remains a frozen retrieval sample with at least one known relevance
error. A fully adjudicated corpus, repeated independent searches and constrained
controller tuning would be new experiments rather than editorial corrections.

## Reproduce the revision

Use the repository's normal environment for historical pipeline reproduction.
The revision was checked with Python 3.12, numpy 2.5.3, scipy 1.18.1, matplotlib
3.11.2, gensim 4.4.0, nltk 3.10.3, scikit-fuzzy 0.5.0 and networkx 3.6.1.
These newer analysis dependencies do not redefine the archived training environment.

From the repository root, with `src` and `assistive-care-flc/src` on `PYTHONPATH`:

```text
python scripts/12_interpretability_panel.py
python scripts/14_seed_validation.py
python scripts/13_report_assets.py
python assistive-care-flc/scripts/run_part3.py
python scripts/15_task2_report.py
python -m pytest tests assistive-care-flc/tests
python paper/latex/check_report.py --doc task1_standalone.tex
python paper/latex/check_report.py --doc task2_standalone.tex
python paper/latex/wordcount.py
```

The panel command reaggregates saved judgments; it does not commission new ones.
The seed command and benchmark command retrain/rerun experiments. The other
report commands derive outputs from saved evidence. Source hashes and pinned
run IDs are in `paper/report_evidence.json`; official CEC provenance is in
`assistive-care-flc/data/cec2005/provenance.json`.

From `paper/latex`, compile `task1_standalone.tex` and `task2_standalone.tex`
with Tectonic or a standard LaTeX installation (repeat compilation to resolve
references). `main.tex` is the combined working copy, not the two-file submission.

The final PDFs are in `output/pdf/`: Task 1 is 9 pages and Task 2 is 17 pages,
including references and appendices. PDF hashes are in `output/pdf/manifest.json`.
The source archive in `paper/output/`
contains the shared LaTeX files and referenced figures for Overleaf; select the
appropriate standalone file as its main document.

## Layout revision (15 September 2026)

Abstracts now use the normal two-column flow. Wide figures are retained, as permitted by IEEE two-column layouts. Removed premature float barriers and page breaks, resized Task 2 Figure 16 to unblock subsequent floats, and placed a barrier before Task 2 references. Figures 19 and 20 now appear on pages 14 and 15 before references. Appendix figures use fixed column-width blocks to preserve numerical order; the MATLAB listing has its own appendix. Final-page columns are balanced in both reports. Rendered and visually checked every standalone page, including the formerly affected regions; no content was removed. Normal unused space below the final content remains.

## Review revision (16 September 2026)

Task 1 Figure 2 uses the user-provided draw.io layout. With user approval,
the duplicated right-branch text was corrected to the natural-sentence,
embeddings, UMAP and HDBSCAN pipeline in `figPipeline_corrected.png`;
the original JPEG is preserved. Repository links appear only in each
report's appendix, with their bibliography entries removed. Contribution
paragraphs use the allocation confirmed by the team, with report preparation
and review shared equally. `paper/TASK1_DEMO_SCRIPT.md` provides a timed
ten-minute narration and app walkthrough, including an honest fallback
when no live provider request is available.

Compiled the standalone reports and `main.tex`; the outputs contain 9, 17
and 26 pages respectively. Reviewed all combined-report pages visually,
including Figure 2 and the final-page flow. Source reference checks pass;
the final compile has no overfull-box warnings. No commit or push was made.

