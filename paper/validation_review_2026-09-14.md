# Assignment and report validation

Reviewed 14 September 2026. This is a review of the existing work, not a rewrite of the submission. The paper, models and saved experimental results have not been changed.

**Assessment: substantial implementation and a useful research question, but the current report is not yet submission-ready.** The highest-value work is to reconcile claims with results, repair compliance gaps, and improve the explanation of the study. More prose and more images alone would not resolve the important weaknesses.

## Evidence and limits of this review

I read the university brief directly, inspected its Task 1 requirements page visually, read the supplied proposal and current LaTeX, and cross-checked decision/experiment records against source code and saved artifacts. Historical `.brain` status tables were treated as records, not proof of the current state.

Verified from the files:

- The frozen corpus contains **1,074 records**, with window counts **8 / 93 / 299 / 674**. The preprocessing report records 1,074 documents and a 3,018-term dictionary.
- The four-model comparison matches its saved evaluation JSON at the quoted precision. The numerical results exist; the main problem is how some of them are described.
- Reaggregating the 39 items from the three saved rater files gives ordinal alpha **0.9786516** and pairwise exact agreement **93.1624%**. Panel means reproduce **4.095 / 2.875 / 2.567 / 2.048** for BERTopic / random search / standard LDA / GA-LDA. This verifies aggregation, not the external provenance or validity of the judgments.
- Recomputing diversity and pairwise Jaccard from saved topic words reproduces all four rows. These lists contain **10 words per topic**, not 25.
- The Task 2 benchmark artifact contains **12 cells and 180 runs**: two functions, two dimensions, three algorithms and 15 runs per cell.
- `check_report.py` passes: 22 Task 1 references, 13 Task 2 references, defined cross-references, existing images and balanced environments. There are **six Task 1 figures and 32 images across the combined document**.
- The selected GA/controller/benchmark tests executed 58 cases: **54 passed; four stability tests failed because SciPy is missing in this environment**. These are dependency failures, not demonstrated algorithm failures. Full model training, the complete test suite and fresh notebook reproduction were not performed.
- The project word counter reports approximately **3,925 Task 1 words**, **7,159 Task 2 body words** and **241 appendix words**. Its exclusions and custom-title handling make these estimates rather than an authoritative submission count.

No current compiled report PDF or local LaTeX engine was available. Consequently, final page count, A4 output, float placement and full report legibility remain unverified. Individual chart assets were inspected, including the temporal plot and co-occurrence network. The supplied proposal PDF has three pages; its historical acceptance should be distinguished from its fit to the brief's one-page instruction.

## 1. University requirements

**Task 1 method and application: substantially met.** LDA topic modelling is an allowed route, and the comparison includes LDA and BERTopic as distinct model families plus two LDA search strategies. A Nepal-affiliated scholarly corpus is a concrete application. Corpus preparation, methods, experimental setup, results, ethical considerations and conclusions are present.

**Individual Contributions: missing from the rendered report.** The entire section is commented out at `paper/latex/main.tex:654`. This is explicitly required by the brief. Restore accurate per-person responsibilities. Do not restore the commented claim that all three members reviewed the labels unless that review actually happened.

**Format and length: unresolved.** The Task 1 deliverable specifies six A4 pages and up to 4,000 words, and identifies a journal template or another single-column format. The current file uses `IEEEtran` conference mode, two columns, without an explicit A4 option. This is a compliance risk unless a tutor has approved a different format. At about 3,925 words, there is no safe basis for simply appending several paragraphs. A six-page layout must be checked after compilation.

**Separate submissions: packaging gap.** The brief requests one file for each task. `main.tex` combines both tasks; `task2_standalone.tex` exists, but there is no corresponding Task 1 entry point in the supplied tree. Prepare two consistently generated submission PDFs and use the required name/student-ID convention. The combined document can remain a working copy.

**Ambiguous overall word limit: do not assume it away.** The administrative block says 4,500 words, while Task 1 separately says 4,000 and Task 2 gives no local cap. The word-count script's interpretation that 4,500 means proposal plus Task 1 is not stated in the brief. Obtain the tutor's applicable interpretation before expanding the already lengthy Task 2 report. Similarly, the brief's dates differ from the recorded extension; retain evidence of the applicable extension. The group-size wording is inconsistent, but the later instructions explicitly allow two or three, so three authors alone is not a demonstrated violation.

**Task 2: broad coverage exists, with a benchmark qualification.** The report and code cover Mamdani design, membership functions, rules, inference, defuzzification, scenarios and surfaces; GA membership-function tuning and chromosome length; and the alternative Sugeno encoding discussion. MATLAB source and screenshot assets are present. Part 3 includes the requested dimensions, repeats, summary statistics and convergence plots. However, its shift vectors are locally generated: these are modified instances of the CEC functions, not the official data instances. The disclosure is good, but it does not establish that this variation satisfies the requirement to use the CEC'2005 suite. Prefer official shift data and a rerun, or retain explicit tutor acceptance of the modification. Also check whether the required benchmark function code must be included in MATLAB form; the supplied MATLAB directory primarily implements the controller.

## 2. Agreement with the submitted proposal

The central project is preserved: Nepal-affiliated AI literature from 2015–2025, LDA, GA optimisation of topic count and priors, BERTopic, coherence/diversity/stability, temporal analysis and LLM-assisted labels. The random-search control is a valuable extension and should stay, including its unfavorable result for the GA.

Four deviations need clear, concise treatment:

1. **Sources:** the proposal names Semantic Scholar as primary, with NepJOL, IEEE and Springer supplementation. The analysed corpus uses OpenAlex. Explain which sources actually contributed records, why the change was made and how it affects coverage. Do not equate implemented but unused clients with collected data.
2. **Peer-review status:** the proposal promises a peer-reviewed, venue-verified corpus. The paper now appropriately says scholarly literature and reports preprint/no-venue records. Journal/conference classification is not itself proof of peer review. For a strict proposal match, perform a documented restricted-corpus sensitivity analysis; otherwise report the departure explicitly.
3. **Temporal method:** separate window refits were replaced by one global model. Given eight records in the earliest window, this is a defensible change. Explain that it measures the changing prevalence of a fixed topic vocabulary, not evolving topic definitions or topic births.
4. **Validated labels:** all 39 topics have generated and automatically reviewed labels. Human validation remains unverified. Use that wording rather than claiming a human-validated topic set.

An important correction to `.brain`: **the supplied `final_proposal.tex` does not explicitly promise three human raters or an inter-rater statistic.** That protocol appears in project planning records. Human review would improve validity, but do not describe its absence as a literal breach of this submitted proposal.

## 3. Corrections that should precede expansion

### A. Correct the two main result claims

At `main.tex:105`, `:452` and `:621`, the claim that GA-LDA improves every metric is false. Standard LDA has lower mean pairwise Jaccard (**0.0359 vs 0.0471**, less topic overlap) and higher automated interpretability (**2.57 vs 2.05**); both assign all documents. List the particular gains instead of claiming a clean sweep.

At `main.tex:111`, `:498` and `:510`, “inverts the coherence ranking” is also too strong. Both C_v and interpretability put BERTopic first and random search second. Only standard LDA and GA-LDA exchange positions:

- C_v: BERTopic > random search > GA-LDA > standard LDA.
- Automated interpretability: BERTopic > random search > standard LDA > GA-LDA.

A defensible central finding is: **GA-LDA improves measured coherence over standard LDA while receiving lower automated interpretability scores.** This is worthwhile without a complete ranking reversal or a claim that coherence optimisation caused lower interpretability.

### B. Repair the evaluation specification

The setup says diversity/Jaccard use top-25 words (`main.tex:397`), but the reporting scripts extract only top-10 lists before passing them to those functions. All saved values reproduce from those ten words. The GA's internal fitness evaluator really does extract enough words for top-25 diversity. Therefore optimisation and final reporting use different diversity definitions.

Either accurately label the current table as D@10/J@10 and disclose D@25 inside fitness, or recompute the reported metrics consistently at 25. The latter requires extracting actual top-25 lists; passing `top_n=25` to a ten-word list cannot recover missing words. Revisit any “wins most metrics” language after this decision.

Perplexity is computed on the corpus used to fit LDA. It is an in-sample variational diagnostic, not demonstrated held-out performance, despite the helper's docstring. Likewise, three stability seeds used during GA selection are reused for its reported score even though the configuration declares a wider final evaluation. Non-optimised metrics on the same corpus are useful diagnostics, not independent validation data.

### C. Narrow single-run search conclusions

One GA run and one random-search run support “random search found the better solution in this matched-budget experiment.” They do not show GA reliability, variance reduction (`main.tex:468`), or that the search strategy generally does not matter. One of 183 random candidates beating GA does not measure between-run variance. Remove these claims or repeat both complete searches across several independent seeds, holding the evaluation budget fixed and reporting distributions.

### D. Synchronise temporal labels and their meaning

The trend table still calls the 0.283 → 0.232 series “Hazard/crop classification”; the audited label for the corresponding GA topic 5 is **Generic ML Evaluation Vocabulary (low coherence)**. It also retains “Speech, text & vision,” although the audited topic 6 label is **Image, Text and Video Recognition**. GA topic 0 is now a mixed low-coherence topic, not a clean regional forecasting theme. This changes the scientific interpretation, not merely terminology.

Build tables, chart legends and prose from `data/processed/final_topic_labels.json`. Retain topic IDs so the mapping can be checked. Show generic/mixed topics honestly; do not give them a substantive domain interpretation just to complete a trend story. The current table presents six of seven topics without clearly identifying the omission.

### E. Make cross-model temporal comparison explicit

`src/nrtm/temporal/proportions.py` removes zero-mass rows before averaging, so BERTopic shares are conditional on assigned documents. LDA shares cover all documents. Report total n, assigned n and outlier fraction for each window, and label the denominator of each plot. These are not directly interchangeable prevalence estimates.

The two families also share the same source corpus and its biases. Agreement cannot prove a finding is “not an artefact of either” (`main.tex:563`). Say the trend is consistent across the two fitted representations. “AI in Education” does not automatically establish that all of its growth is generative AI; inspect representative documents before applying the narrower label. The apparent generative-AI share in 2015–2017 particularly needs a document check, since retrospective assignment to a broad education topic is not evidence of modern generative-AI research at that time.

### F. Correct figure provenance and improve legibility

`scripts/09_figures.py` feeds processed `tokens.jsonl` into the word cloud and co-occurrence graph. The graph is not an image of raw, uncleaned text as `main.tex:304` states. The word cloud uses document frequency, which should be named. Claims that those pictures demonstrate removal of data/accuracy/performance are also problematic: the configuration explicitly retains those terms.

The co-occurrence graph has overlapping labels and a dense mesh of edges; it communicates little about the research question at single-column size. Its before-cleaning rationale is not supported by the generating script. Replace it with a measured preprocessing comparison or a directly relevant topic-quality figure. The temporal plot's long keyword legend consumes substantial width; use validated short labels, topic IDs, direct labels or small multiples. Show the n=8 window with hollow markers or a distinct background and avoid presenting its connecting segment as strong trend evidence.

### G. Be precise about automated assessment and disclosure

State **three blinded LLM judgments** in the abstract, not only later in Methods. Separate contexts do not establish independent measurement error when judges use the same underlying model. High agreement establishes consistency of recorded ratings; it does not establish human interpretability or label correctness. Give model/version as recorded, prompt/rubric, date and sampling settings where known; mark unavailable settings honestly.

The sentence at `main.tex:602` stating no language model was used to write the submission needs reconciliation with project records that describe paper drafting and rewriting. Replace it with an accurate, team-confirmed account of assistance and authors' verification. This is a concrete inconsistency in the current report, not a conclusion about misconduct.

### H. Fix reproducibility overclaims

`scripts/12_interpretability_panel.py:26` hard-codes another computer's repository path. The notebook defaults to a quick mode that explicitly does not reproduce final numbers, and some evaluation cells extract ten words while labelling metrics as top-25. Do not describe a quick-mode execution as full result reproduction. Fix portable paths and document the exact full configuration, run IDs and which steps load fixed judgments rather than rerun them. The public repository's current completeness was not verified because web retrieval failed.

## 4. How to implement the feedback without padding

**Introduction: target roughly 400–500 words, by redistributing the existing budget.** The opening currently moves into LDA mechanics before establishing why this regional analysis matters. Use five connected paragraphs:

1. The practical problem: researchers and institutions need an interpretable map of a fragmented literature, not just publication counts.
2. Scope: define Nepal affiliation rather than nationality, what title/abstract records reveal, and what excluded local outlets mean.
3. Methodological challenge: a small, heterogeneous corpus, sensitive topic granularity, and the trade-off between complete assignment and specific topics.
4. Study questions: regional themes, changing prevalence, and the value of optimisation relative to a matched control and BERTopic.
5. Contributions: curated corpus, reproducible comparison and a qualified temporal account; make no unsupported first-study claim.

Move Dirichlet-prior explanation and pipeline internals into Methods. Replace categorical claims such as “Nepal's AI output has no systematic account” with a precisely supported gap statement.

**Related Work: promote it to its own section, roughly 350–450 words.** Organise it around regional scientometrics, search-based topic-model tuning, and evaluation validity. For each thread explain the earlier evidence, its limitation and the design decision it motivates here. Do not turn this into a chronological list of papers. Use actual dataset sizes and evaluation settings where verified, rather than broad claims that previous corpora were all large, clean or homogeneous.

One verified bibliography correction: the cited AI-trends paper is by **Dejian Yu and Bo Xiang**, not J. Yu and Y. Xiang. The author's publication page also describes country/region aggregation, so the contrast cannot simply be global work versus no regional analysis. See the [author's record](https://li-pohsiang.github.io/) and [publisher record](https://www.sciencedirect.com/science/article/abs/pii/S0957417423006164). The agriculture reference exists, but the current entry omits authors, year and part of its title; complete it from the [publisher record](https://www.mdpi.com/2624-7402/8/8/312). This was a targeted check, not a complete reference audit.

**Add missing explanatory substance elsewhere:** give a compact parameter specification (actual K/prior ranges, passes/iterations, seeds, topic-word cutoffs, fitness normalisation), a topic case study with representative document evidence, and a short explanation of what low stability permits one to conclude. Clarify that the GA scalarises multiple objectives through a weighted sum rather than estimating a Pareto frontier.

**Recover space:** shorten the unusually detailed abstract to about 180–220 words, remove repeated declarations of GA defeat and ranking inversion across Results/Discussion/Conclusion, and replace lengthy operator justifications with one concise paragraph or parameter table. Keep one interpretation of each result. There is room for a much stronger introduction and review without increasing the total substantially. Any net expansion beyond 4,000 needs an explicitly revised limit.

## 5. Figure plan, in priority order

1. **End-to-end method diagram — add.** Show collection and filtering, the shared corpus, the two preprocessing branches, the three LDA strategies and BERTopic, evaluation and temporal reporting. This explains the entire study in one view. No new experiment needed.
2. **Coherence versus automated interpretability — add.** Four clearly labelled model points using the existing means. Caption it as a model-level descriptive comparison, not a correlation study or evidence of causality. Explain that BERTopic and random search retain their rankings.
3. **Paired temporal panels — expand the current figure.** GA-LDA and BERTopic with audited labels, aligned windows, assigned/total n and an explicitly qualified earliest window. BERTopic temporal artifacts already exist. A heatmap may be more legible than fourteen lines.
4. **Search efficiency at equal evaluation budget — desirable.** GA and random-search incumbent fitness against cumulative unique evaluations. Use saved histories only if candidate ordering and counts support the alignment. The existing GA generation plot alone does not show the control. Multi-run uncertainty bands require new runs; do not fabricate them from one history.
5. **Preprocessing evidence — replacement option.** Measured before/after document frequencies for a few removed and retained terms, with both stages actually computed. Replace the dense co-occurrence graph, and consider dropping the word cloud if space is tight.

Aim for seven or eight useful figures/panels only if they fit the approved layout. Keep the yearly distribution and K-sweep. Prefer vector exports and assess labels at final print size. Every added figure should answer a research question or explain a reproducible decision and receive an analytical sentence in the text.

## 6. Work order

**First, corrections using existing evidence:** restore contributions; correct metric and ranking statements; reconcile top-word cutoffs; apply audited labels; disclose automated judgments clearly; fix figure provenance and portable paths; document proposal deviations. These do not require collecting a new corpus.

**Second, restructure the report:** strengthen the introduction, separate Related Work, improve Methods precision, replace weak figures, and cut repetitive conclusions. Work within the verified limit and produce separate Task 1/Task 2 PDFs.

**Third, strengthen evidence if time allows:** repeat complete GA/random searches over independent seeds; evaluate final configurations on additional unused seeds; perform a shared-vocabulary coherence sensitivity check; collect actual human ratings using the existing blind pack; inspect representative papers for temporal claims; rerun Task 2 with official benchmark shifts. Keep these explicitly separate from editorial fixes.

**Finally, validate the exact submission:** compile and inspect every page, verify word count under the applicable counting rules, check numbers and captions against pinned artifacts, run the full notebook in the declared environment, verify the accessible code link, and ensure contribution/disclosure statements are accurate.

The report's strongest defensible story is a careful regional application showing that different measures favor different topic models, with a useful matched-budget control. The current sweeping claims obscure that contribution. Correcting them will make the report more persuasive, not less.
