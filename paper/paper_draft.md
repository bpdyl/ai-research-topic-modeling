# Optimized Topic Modeling of AI Research Involving Nepali Authors: Tracking Thematic Evolution and Trends in Nepal (2015–2025)

**Bibek Paudyal (250288) · Siddhartha Bhatta (250620) · Sajan Mahat (250289)**
ST7085CEM Advanced Machine Learning · Softwarica College of IT & E-Commerce (Coventry University)

> **DRAFT — v2, 2026-09-06.** Numbers are final and traceable to `EXPERIMENT_LOG.md`
> (EXP-001…EXP-008). Revised after the budget-matched random-search control (EXP-008) **reversed
> the paper's central claim about the genetic algorithm**. Two items outstanding, marked `[TODO]`:
> human interpretability ratings and the individual-contributions allocation.

---

## Abstract

Artificial intelligence research produced in low-resource countries is rarely characterised
systematically, and topic-model comparisons are typically run on large, clean English corpora
unlike the regional literature most institutions actually hold. We construct a corpus of 1,074
peer-reviewed and preprint AI/ML papers (2015–2025) with at least one Nepal-affiliated author and
compare three topic models on it: standard Latent Dirichlet Allocation, a genetic-algorithm-optimised
LDA searching *(K, α, β)* against a multi-objective fitness of coherence, diversity and stability,
and BERTopic. GA-optimised LDA improves on the grid-searched baseline across every computed metric, including
three it never optimised — c_npmi (−0.0230 → 0.0112), u_mass (−3.36 → −2.18) and perplexity
(172.1 → 167.8). **However, a budget-matched random search over the same space beat it**
(fitness 0.5282 vs 0.5196; C_v 0.5030 vs 0.4940), so the gain is attributable to searching
*(K, α, β)* finely rather than to the evolutionary mechanism. We report this negative result because
without the control the opposite conclusion would have looked well supported. Two further findings
qualify the comparison. First, **no model dominates**: BERTopic attains the highest C_v (0.5107) and
a far higher cross-seed stability (0.842 vs 0.338) while leaving 17.1% of documents unclustered.
Second, an interpretability assessment **inverts** the coherence ranking — GA-LDA, which won most
numeric metrics, scores *lowest* (2.71/5) because the optimiser converged to K=7 and produced broad,
generic topics, against BERTopic's 4.64/5. Both model families independently identify generative AI
in education as the dominant rising theme (+0.131 and +0.181 share). We conclude that optimising
coherence metrics optimises neither interpretability nor, relative to random search, anything the
evolutionary machinery uniquely provides.

**Keywords:** topic modelling, LDA, genetic algorithms, BERTopic, scientometrics, Nepal

---

## 1. Introduction

AI research output has grown faster than any individual can track by reading. Scientometric topic
modelling addresses this by inferring themes from large publication corpora, and Latent Dirichlet
Allocation (LDA) [2] remains the standard tool because its topics are directly interpretable as
word distributions.

Two gaps motivate this study. First, the scientometric literature concentrates on global corpora;
research produced in low-resource countries is characterised anecdotally if at all, and Nepal's AI
output has no systematic account. Second, methodological comparisons of classical against neural
topic models are typically conducted on large, homogeneous corpora — precisely the conditions under
which LDA is best behaved. Whether an evolutionary-optimised LDA can match a modern embedding model
on a *small, heterogeneous, regional* corpus is a different and less-studied question.

### 1.1 Related work

Agrawal, Fu and Menzies [1] show that LDA's instability under random initialisation is severe and
that metaheuristic search over its hyperparameters materially improves both stability and coherence.
Their argument motivates our proposed model directly, and we test their claim on a corpus type they
did not study. Yu and Xiang [2] apply standard LDA to global AI literature, establishing the
thematic-mapping baseline this work extends temporally and regionally. Embedding-based topic
modelling with BERTopic has been applied at scale to domain literature [3], reporting more specific
topics than LDA but without the document-level topic distributions LDA provides. Haq et al. [4]
apply text mining to 200 cloud-computing papers, producing a single static snapshot — the limitation
this study addresses by tracking themes across four time windows.

### 1.2 Contribution

The contribution is the combination rather than any single component: a regional scientometric
application, a GA-optimised LDA with a multi-objective fitness, and a three-way evaluation that
includes interpretability alongside coherence. We do not claim to invent an algorithm or to be first
to optimise LDA with a GA. We report two findings we did not expect, both negative for our own
proposed model: a budget-matched random search **outperforms** the genetic algorithm, and the
interpretability ranking **inverts** the coherence ranking.

---

## 2. Problem and dataset

### 2.1 Problem

We ask four questions. **RQ1:** what themes characterise AI research involving Nepal-affiliated
authors, 2015–2025? **RQ2:** how have those themes shifted across the decade? **RQ3:** does
GA-optimised LDA outperform grid-searched LDA? **RQ4:** how does it compare with BERTopic?

### 2.2 Corpus construction

Papers were retrieved through the **OpenAlex** works API, filtered server-side on
`authorships.countries:np` together with AI/computer-vision topic subfields and a publication-year
range, then supplemented by a keyword pass to recover applied AI filed under other primary topics.

**Deviation from the proposal, stated explicitly.** The proposal named Semantic Scholar as the
primary source with NepJOL, IEEE Xplore and Springer as supplements. OpenAlex was used instead, for
a methodological reason: it exposes a structured `authorships.countries` filter, giving a reliable
signal for "has a Nepal-affiliated author", whereas Semantic Scholar's bulk-search endpoint cannot
return author affiliations at all (requesting them returns HTTP 400), which would have forced
fragile institution-name string matching. NepJOL has no public API and OJS table-of-contents
scraping was judged out of scope; it is a genuine limitation of coverage, likely under-representing
Nepali-language and locally-published work. IEEE and Springer clients were implemented but not
enabled. We regard the substitution as strengthening the proposal's stated intent — a structured
country filter is a better instrument for the inclusion rule than free-text matching — but it is a
deviation and is reported as one.

### 2.3 Inclusion and filtering

Inclusion required ≥1 author with a Nepal-based institutional affiliation, and AI-method vocabulary
in the title or abstract. From 1,693 retrieved records the analysis corpus was built by a documented
funnel:

| Filter | Rationale | Remaining |
|---|---|---:|
| — | retrieved | 1,693 |
| `year ≤ 2025` | match the stated 2015–2025 span; 2026 records are disproportionately preprints | 1,228 |
| abstract required | title-only documents are too short to model | 1,102 |
| ≥30 tokens | removes editorials and commentaries | 1,088 |
| exclude "computer vision syndrome" | see below | **1,074** |

The last filter deserves note. The AI-relevance gate matches the phrase *computer vision*, which
also occurs in **Computer Vision Syndrome** — an ophthalmic condition caused by screen use. Fourteen
occupational-health papers entered the corpus this way and, at K=10, formed their own topic. All
fourteen were inspected manually; none contained AI-method vocabulary. This was found by reading
model output, not by auditing the corpus, which is itself an argument for inspecting topics before
trusting metrics.

### 2.4 Corpus properties

1,074 documents; 564 distinct venues; median document length 212 words. **93.5% are journal or
conference publications**; 3.4% are preprint-server records (SSRN, arXiv, Zenodo, Research Square)
and 3.1% carry no venue. We therefore describe the corpus as *scholarly AI literature* rather than
claiming it is wholly peer-reviewed.

The temporal distribution is heavily skewed: **8 / 93 / 299 / 674** documents across 2015–17,
2018–20, 2021–23 and 2024–25. This skew constrains the analysis design (§3.5) and every temporal
claim.

A 30-record manual audit found **0 incorrect Nepal affiliations** (100% precision on the sample).
The audit did reveal a reporting defect in our own affiliation-extraction field, which matched only
the literal string "Nepal" and so under-reported affiliations such as "Tribhuvan University";
corrected, 99.8% of records carry an identifiable Nepali institution. The audit also found ~7% of
sampled papers are law or policy papers *about* AI rather than papers *using* AI methods — a
borderline category we retain, since scientometric studies of AI conventionally include AI policy
and ethics.

---

## 3. Methods

### 3.1 Preprocessing

Documents are `title + abstract`. Cleaning applies NFKC normalisation, strips leading "Abstract"
prefixes (71 documents), copyright tails, URLs, DOIs and parenthesised acronym glosses. That last
step matters: authors write "artificial intelligence (AI)", and phrase detection consequently
produced *both* `artificial_intelligence` (191 documents) and `artificial_intelligence_ai` (160) for
one concept. Stripping the gloss consolidated it to 265 while leaving standalone `ai` (253) intact.

For LDA, text is lowercased, tokenised, stop-word filtered, lemmatised (WordNet, noun then verb),
and bigram/trigram phrases detected. **Technical acronyms are protected** from the minimum-length
filter and the lemmatiser, so `ai`, `ml`, `nlp`, `cnn`, `lstm` survive; 22 of 29 configured acronyms
reach the final dictionary, the remainder falling below the document-frequency floor.

Domain stop words were selected by **measurement, not intuition**. A candidate list of generic
research-process vocabulary was scored by document frequency, and the 77 terms at DF ≥ 5% removed —
*challenge* 32.5%, *enhance* 30.4%, *include* 30.2%, *improve* 29.0%. The standard `no_above=0.5`
filter caught only one term ("model", 59.8%); everything in the 18–33% band would otherwise have
appeared in every topic. Two candidates, `level` and `value`, were deliberately **retained**: this
corpus contains hydrology and health research where "water level" and "p-value" are content.

Final dictionary: **3,018 terms** after `no_below=5, no_above=0.5`; 0 empty documents.

**BERTopic consumes a separate branch** — cleaned natural sentences, not the LDA token stream. A
sentence transformer depends on word order and function words; supplying it lemmatised bags of words
would have crippled the benchmark and invalidated the comparison.

### 3.2 Standard LDA (baseline)

Gensim `LdaModel`, K swept over {5,10,…,40}, selected by C_v. The comparison is therefore
GA-search versus grid-search, not GA versus an arbitrary K.

### 3.3 GA-optimised LDA (proposed)

A genetic algorithm written from scratch searches *(K, α, η)*.

- **Encoding:** real-valued, not binary. α and η span two orders of magnitude; a bit-string would
  impose an arbitrary discretisation on the dimensions the search exists to explore finely.
- **Initialisation:** log-uniform on α and η. Uniform sampling over [0.01, 1.0] places ~90% of draws
  above 0.1 and barely explores the sparse-prior region where topic models typically operate.
- **Selection:** tournament (size 3). Fitness occupies a narrow band (~0.40–0.52), so
  fitness-proportionate selection would apply near-equal pressure to good and bad genomes.
- **Crossover:** BLX-α (rate 0.7). Plain interval crossover can only interpolate between parents, so
  the population contracts toward its own mean and cannot reach an optimum outside the initial spread.
- **Mutation:** Gaussian, σ = 0.15 × gene range (rate 0.05). Scaling σ per gene keeps mutation
  comparable across parameters whose ranges differ by orders of magnitude.
- **Elitism:** 2. Without it the best solution can be lost and the convergence curve is not monotonic.
- **Fitness:** `0.6·norm(C_v) + 0.2·diversity + 0.2·stability`, C_v min-max normalised over
  [0.30, 0.70] so all three terms share a scale.

Population 20, 15 generations. Fitness is cached on the rounded genome; the cache achieved a
**39% hit rate**, reducing 900 LDA fits to 549.

### 3.4 Random-search control

The GA was compared against a grid, and the grid tested 8 values of K with fixed α and η while the
GA searched a continuous 3-D space. Any advantage could therefore be explained either by the
evolutionary mechanism or simply by searching more finely. To separate them we ran random search
over the identical space with the identical fitness function, stability seeds and LDA settings, at a
**budget matched on unique fitness evaluations** (183, taken from the GA's actual cost — its 39%
cache hit rate meant 300 scheduled evaluations cost 183 real ones). Both strategies draw candidates
from the same log-uniform sampler. The two runs differ in exactly one respect: how candidates are
proposed.

### 3.5 Stability

The proposal specifies stability without defining it. We define it as: fit the configuration under
*m* seeds; for each pair of runs, match topics one-to-one by maximising total Jaccard overlap of
top-N word sets (Hungarian assignment); take the mean matched overlap, averaged over pairs. Hungarian
matching is essential because topic indices are arbitrary — two runs recovering identical topics in
different orders would otherwise score near zero.

### 3.6 Temporal analysis

The proposal specified re-fitting the best model per time window. **This is not executable**: the
2015–17 window holds 8 documents. We instead fit one global model and report mean document–topic
proportions per window, normalised *within* window so window size cancels. The 2015–17 window is
reported but excluded from trend fitting (<30 documents), and per-window *n* is printed on every
table and figure.

### 3.7 Evaluation

C_v (primary), c_npmi and u_mass; topic diversity and mean pairwise Jaccard at top-25; stability;
perplexity; and interpretability (§4.3). Reporting three coherence measures is deliberate: the GA
optimises C_v, so quoting C_v alone as evidence of its success would be circular.

---

## 4. Results

### 4.1 Model comparison

| Metric | Standard LDA | GA-Optimized LDA | Random-Search LDA | BERTopic |
|---|---:|---:|---:|---:|
| Topics (K) | 10 | 7 | 8 | 14 |
| Coherence C_v | 0.4483 | 0.4940 | 0.5030 | **0.5107** |
| Coherence c_npmi | −0.0230 | 0.0112 | **0.0245** | −0.0300 |
| Coherence u_mass | −3.3586 | **−2.1796** | −2.5477 | −3.4444 |
| Topic diversity | 0.7900 | **0.8000** | **0.8000** | 0.7286 |
| Mean pairwise Jaccard | **0.0359** | 0.0471 | 0.0536 | 0.0471 |
| Stability (3 seeds) | 0.2526 | 0.3375 | 0.3382 | **0.8418** |
| Perplexity | 172.1 | 167.8 | **165.9** | — |
| Unmodelled documents | **0%** | **0%** | **0%** | 17.1% |
| **Interpretability (1–5)** | 3.10 | 2.71 | [TODO] | **4.64** |

**RQ3 — GA-LDA versus baseline, and versus a matched control.** Against the grid-searched baseline
GA-LDA improves every computed metric. The C_v gain (+10.2%) is partly expected since C_v is the
optimisation target; more meaningful are c_npmi, u_mass and perplexity, **none of which the GA saw**.
c_npmi crossing from negative to positive is substantive: negative c_npmi indicates top words close
to statistical independence.

**But the random-search control reverses the conclusion.** At an identical budget of 183 unique
evaluations, random sampling of the same space found a *better* solution:

| | Random search | GA | GA advantage |
|---|---:|---:|---:|
| best fitness | **0.5282** | 0.5196 | −0.0086 |
| C_v | **0.5030** | 0.4940 | −0.0091 |
| c_npmi | **0.0245** | 0.0112 | −0.0134 |
| u_mass | −2.5477 | **−2.1796** | +0.3681 |
| diversity | 0.8000 | 0.8000 | 0.0000 |
| stability | 0.3382 | 0.3375 | −0.0007 |

The GA wins only u_mass, a measure neither strategy optimised. **We therefore cannot claim that
evolutionary search outperforms alternatives on this problem.** The supportable claim is narrower:
searching *(K, α, β)* finely beats an 8-point grid, and the search strategy is not what matters.

One nuance makes the result less damning than the headline suggests. Only **1 of 183** random draws
beat the GA's final solution. The GA reliably reaches a good region; random search needed 183
attempts to find one better point. On this evidence the GA behaves as a *variance-reduction*
mechanism rather than an optimum-finding one — valuable if a single run must be trusted, but not
what its use here was premised on.

**RQ4 — versus BERTopic.** No model dominates. BERTopic takes C_v and stability by large margins but
is worst on c_npmi, u_mass and diversity, and leaves 17.1% of documents unclustered. Two caveats
qualify its coherence figures: its topics derive from 83% of the corpus, and only 66.4% of its topic
words exist in the shared reference dictionary, the remainder being dropped before scoring.

BERTopic's stability advantage (0.842 vs 0.338) has a clear mechanism: embeddings are deterministic,
so only UMAP and HDBSCAN vary across seeds, whereas LDA's variational inference is re-initialised
entirely.

### 4.2 Convergence

Best fitness rose 0.4865 → 0.5196 by generation 8 and was **flat thereafter**; population mean rose
0.4140 → 0.5130, nearly reaching the best. The population converged onto one genome and roughly half
the compute produced no improvement.

This is the most likely explanation for §4.1's control result. From generation 8 onward the GA was
resampling a narrow neighbourhood while random search continued to explore the whole space, so the
GA effectively spent half its budget on a smaller random search. A higher mutation rate, a diversity
maintenance mechanism, or restarts would be the obvious remedies, and testing them is the clearest
line of future work.

### 4.3 Interpretability — and an inverted ranking

Each topic was rated 1–5 on whether its top-10 words name one identifiable research area.
**BERTopic 4.64 (100% rated ≥4) > standard LDA 3.10 > GA-LDA 2.71 (14% rated ≥4).**

**This inverts the coherence ranking, and it is the paper's most consequential finding.** The GA
converged to K=7 because fewer, broader topics scored well on a fitness of coherence + diversity +
stability. Those topics are correspondingly generic: GA-LDA topic 5 is
`accuracy, data, performance, classification, prediction, image, detection, dataset` — entirely
evaluation vocabulary, describing a method rather than a subject. BERTopic's topics name domains
directly: `drug, molecular, inhibitors, screening, binding`; `concrete, compressive strength`;
`stock, price, market, trading`.

**Optimising coherence metrics did not optimise interpretability; on this corpus it worked against
it.** A study reporting only C_v would have concluded GA-LDA was the better model.

> **[TODO — human ratings]** These ratings come from a **single automated rater**, not the three
> independent human raters the proposal specifies. Inter-rater agreement is therefore not reported.
> The same system fitted the models and rated their output, which is a conflict of interest we state
> rather than conceal. A blind, shuffled rating instrument is prepared; human ratings, once
> collected, supersede these numbers.

### 4.4 Thematic evolution (RQ1, RQ2)

| GA-LDA topic | 2015–17 (n=8) | 2018–20 (n=93) | 2021–23 (n=299) | 2024–25 (n=674) | trend |
|---|---:|---:|---:|---:|---|
| Generative AI in education | 0.108 | 0.021 | 0.064 | **0.152** | **rising (+0.131)** |
| Construction & industrial analytics | 0.017 | 0.102 | 0.120 | 0.138 | rising |
| Energy infrastructure & security | 0.052 | 0.158 | 0.147 | 0.171 | rising |
| Applied ML for hazard/crop classification | 0.283 | 0.276 | 0.260 | 0.232 | declining |
| Speech, text & vision recognition | 0.256 | 0.183 | 0.174 | 0.136 | declining |
| Regional forecasting & control | 0.138 | 0.180 | 0.130 | 0.094 | declining |

BERTopic, fitted independently on a different representation, agrees: `AI in education`
0.014 → 0.195 (+0.181), `AI in healthcare` +0.075, with `Nepali language processing`,
`power systems` and `COVID-19 modelling` declining. **A finding reproduced across two model
families is not an artefact of either.**

The corpus profile answers RQ1: Nepal's AI research is dominated by *applied* work — agriculture,
hydrology and hazards, health informatics, construction — rather than core ML methodology.
Nepal-specific themes appear directly, including landslide susceptibility mapping and Nepali-language
NLP.

**This only partly confirms our proposal's prediction.** We expected a shift toward "deep learning,
transformer models and generative AI". The generative-AI half is strongly supported. The
deep-learning half is **not**: deep-learning topics *decline* as a share. Our interpretation is
saturation — deep learning became the default method rather than a distinct theme — but that is
interpretation, not measurement. Shares are relative: a declining share does not mean less absolute
research, since the corpus grows sharply.

---

## 5. Social, ethical, legal and professional considerations

**Scientometric profiling of identifiable people.** This corpus names institutions and, through
author lists, individual researchers in a small national research community where a topic maps to a
handful of people. Aggregate thematic findings are legitimate; ranking named institutions or
individuals would not be, and we do not. We report themes, never productivity by person or institution.

**Data provenance and licensing.** OpenAlex data is CC0 and its API was used within its documented
polite-pool conventions with rate limiting and a contact email. Only metadata and abstracts were
retrieved — no paywalled full texts.

**Equity in representation.** English-language, Western-indexed databases systematically
under-represent scholarship published locally and in Nepali. NepJOL, the national journal platform,
is *absent* from this corpus for the technical reason that it has no public API. Our picture of
"AI research in Nepal" is therefore a picture of *internationally-indexed* AI research involving
Nepali authors, and the gap is likely to fall on exactly the locally-oriented work a study of Nepal
should most want to capture.

**LLM use, declared.** A language model drafted candidate topic labels from top-word lists and
produced the interpretability ratings of §4.3. This is an instrument used in the method, and it is
recorded reproducibly — the exact prompt template, the model, and the pending human-review status
are all in the repository. It is distinct from using a language model to write the submission, which
we have not done. Because the same system fitted the models and rated their output, we treat those
ratings as provisional pending human review.

**Professional responsibility in reporting.** Our proposed model won most numeric metrics and lost
on interpretability. Reporting only the metrics that favoured it would have been the easier paper
and a misleading one.

---

## 6. Discussion and conclusions

**Metric choice can reverse a conclusion.** GA-LDA wins 4 of 9 numeric rows and loses
interpretability decisively. Optimising coherence drove K down and produced broader, less meaningful
topics. Practitioners selecting a topic model on coherence alone should expect this failure mode.

**Hyperparameter search works; the evolutionary machinery is not what makes it work.** GA-LDA beats
the grid-searched baseline on every metric, including three it did not optimise. But a random search
at matched budget beat the GA in turn. The honest conclusion is that LDA on this corpus is
sensitive to *(K, α, β)* and rewards any fine search, and that our proposed mechanism contributed
nothing detectable beyond that.

We report this because the alternative was tempting and wrong. Without the control, EXP-002 alone
would have supported "GA-optimised LDA outperforms standard LDA" — true as stated, and misleading as
an argument for genetic algorithms. A three-dimensional continuous space is precisely the regime in
which random search is known to be a strong baseline, and our GA's premature convergence
(§4.2) removed the exploration that might have justified the extra machinery.

**Embeddings are markedly more stable on a small corpus** (0.842 vs 0.338). For a 1,074-document
corpus this is a strong practical argument for BERTopic — though 17.1% of documents go unmodelled,
which is unacceptable where document-level assignment is required, as in our temporal analysis.

**Limitations.** (i) Absolute stability for every LDA variant is low (0.25–0.34): even the best
model recovers about a third of the same structure across seeds, which qualifies every topic-level
claim here. (ii) The 2015–17 window holds 8 documents; its trend line is indicative only. (iii)
NepJOL's absence biases coverage against locally-published work. (iv) Interpretability rests on one
automated rater, and the random-search model has not been rated at all. (v) The GA was run once with
one parameter setting; a fairer test of evolutionary search would tune its mutation rate and
population size, and repeat both searches over multiple seeds to compare distributions rather than
single runs.

**Conclusion.** On a small regional corpus, fine hyperparameter search improves LDA measurably over
grid search, but our genetic algorithm achieved nothing that random sampling at the same cost did
not. Meanwhile the model that best serves the actual research purpose — reading what a field is
about — is BERTopic, which the coherence metrics ranked ambiguously and the interpretability
assessment ranked first. Both of our headline methodological expectations were wrong in the same
direction: the metrics we optimised were not measuring what we cared about. Both approaches agree that generative AI in
education is the fastest-rising theme in Nepal-affiliated AI research, and that the field is
predominantly applied rather than methodological.

---

## 7. Individual contributions

> **[TODO]** Proposed allocation below, **to be confirmed and edited by the team before submission**.
> It is inferred from repository evidence (Bibek Paudyal authored the scraper commit and owns the
> GitHub remote) and from the team split recorded during planning. It must not be submitted unverified.

- **Bibek Paudyal (250288)** — corpus acquisition: OpenAlex/Semantic Scholar clients, Nepal-affiliation
  and AI-relevance filtering, deduplication, corpus validation.
- **Siddhartha Bhatta (250620)** — modelling: LDA baseline and K sweep, genetic algorithm
  (encoding, operators, fitness), BERTopic.
- **Sajan Mahat (250289)** — evaluation and analysis: coherence/diversity/stability harness, temporal
  analysis, topic labelling, figures, manuscript.

---

## References

[1] A. Agrawal, W. Fu and T. Menzies, "What is wrong with topic modeling? (and how to fix it using
search-based software engineering)," *Information and Software Technology*, vol. 98, pp. 74–88, 2018.

[2] J. Yu and Y. Xiang, "Discovering topics and trends in the field of artificial intelligence: Using
LDA topic modeling," *Expert Systems with Applications*, 2023.

[3] "Mapping the evolution of artificial intelligence in agriculture: A large-scale BERTopic
analysis," *AgriEngineering*, vol. 8, no. 8, art. 312, MDPI.

[4] M. I. U. Haq, Q. Li and S. Hassan, "Text mining techniques to capture facts for cloud computing
adoption and big data processing," *IEEE Access*, vol. 7, pp. 162254–162267, 2019.

[5] M. Grootendorst, "BERTopic: Neural topic modeling with a class-based TF-IDF procedure,"
*arXiv:2203.05794*, 2022.

[6] M. Röder, A. Both and A. Hinneburg, "Exploring the space of topic coherence measures," in *Proc.
8th ACM Int. Conf. Web Search and Data Mining (WSDM)*, 2015, pp. 399–408.

[7] J. Chang, J. Boyd-Graber, S. Gerrish, C. Wang and D. Blei, "Reading tea leaves: How humans
interpret topic models," in *Advances in Neural Information Processing Systems 22*, 2009.

[8] A. B. Dieng, F. J. R. Ruiz and D. M. Blei, "Topic modeling in embedding spaces," *Transactions of
the ACL*, vol. 8, pp. 439–453, 2020.

---

## Appendix A — Reproduction

Code: `https://github.com/bpdyl/ai-research-topic-modeling` (branch `core-development`).

```
pip install -r requirements.txt
python -c "import nltk; [nltk.download(r) for r in ['punkt_tab','stopwords','wordnet','omw-1.4']]"
python scripts/01_build_corpus.py     # 1,693 -> 1,074, corpus report
python scripts/02_preprocess.py       # tokens, dictionary (3,018), BoW/TF-IDF
python scripts/03_lda_baseline.py     # EXP-001
python scripts/04_ga_lda.py           # EXP-002  (~69 min)
python scripts/05_bertopic.py         # EXP-003
python scripts/06_evaluate.py         # EXP-004  + blind rating sheet
python scripts/07_temporal.py         # EXP-005
python scripts/08_label_topics.py     # EXP-006
python scripts/10_interpretability.py # EXP-007
python scripts/11_random_search.py # EXP-008  (~66 min, budget-matched control)
python scripts/09_figures.py
```

Every run writes `results/runs/<RUN_ID>/` containing the exact config used, the git commit, and its
metrics. All seeds are fixed at 42. 69 unit tests cover corpus filtering, preprocessing, acronym
preservation and every GA operator.

**Environment note.** gensim's `CoherenceModel` spawns one worker per core by default; on Windows
each child re-imports the full scipy/BLAS stack, which exhausted the system commit limit during
development. Coherence is pinned to a single process and BLAS thread pools are capped at import time.

## Appendix B — Figures

`fig_corpus_by_year.png` · `fig_wordcloud.png` · `fig_cooccurrence.png` ·
`fig_coherence_vs_k.png` · `fig_ga_convergence.png` · `fig_trends_ga_lda.png`
