# Project Briefing — ST7085CEM Advanced Machine Learning

**Team:** Bibek Paudyal (250288) · Sajan Mahat (250289) · Siddhartha Bhatta (250620)

---

## Overview

The coursework has two independent tasks, each written up as its own paper.

| | **Task 1** | **Task 2** |
|---|---|---|
| **Topic** | What AI research involving Nepali authors is about, and how it changed over 2015–2025 | An automatic heating and lighting controller for a flat where a person with limited mobility lives |
| **Methods** | Topic modelling: LDA, genetic-algorithm-tuned LDA, random search, BERTopic | Fuzzy logic control, genetic-algorithm tuning, and a comparison of three optimisers |
| **Data** | 1,074 research papers from OpenAlex | Simulated sensor data, plus two standard benchmark functions |
| **Key result** | The tuned model scored best on the numbers but was the hardest for readers to interpret | Tuning made the controller 32.6% more accurate but broke two safety-relevant properties |

---

## Task 1 — Topic Modelling of AI Research Involving Nepali Authors

### What we set out to do

Nobody had mapped AI research in Nepal systematically. We used **topic modelling**, which
finds themes in a large collection of text by noticing which words tend to appear together, to
answer two questions. What are the main research themes, and how have they shifted over time?
And does an automatically tuned topic model actually beat the standard one?

### Data

- **1,074 AI/ML papers (2015–2025)** with at least one Nepal-affiliated author, collected from
  OpenAlex (an open index of scholarly publications).
- Filtered from 1,693 retrieved records: kept papers up to 2025 that have an abstract and at
  least 30 words, and removed 14 medical papers about "computer vision syndrome" (eye strain)
  that the keyword search had wrongly picked up.
- The text was cleaned before modelling: generic research words such as "data", "model" and
  "improve" were removed based on how often they appear, while technical acronyms such as
  AI, NLP and CNN were kept.

### Methods — four models compared

| Model | What it is |
|---|---|
| **Standard LDA** | The classic topic model. The number of topics (K) was chosen by trying 5, 10, … 40; K = 10 scored best |
| **GA-optimised LDA** | LDA whose settings (K, α, β) are tuned by a **genetic algorithm**, a search that mimics evolution by keeping good candidates, combining them and adding small random changes |
| **Random-search LDA** | The same search with the same budget (183 tries), but with settings picked at random. This control tells us whether the genetic algorithm itself adds anything |
| **BERTopic** | A modern neural approach: a language model turns each paper into a numerical "meaning" vector, and similar papers are grouped together |

**How we judged them:** *coherence* (do a topic's words really appear together?), *diversity*
(are the topics distinct?), *stability* (do repeated runs give the same topics?) and
*interpretability* (can a reader name each topic?). Three independent, blind raters scored
interpretability from 1 to 5. The raters were AI language models, which the paper states
openly.

### Results

| | Standard LDA | GA-LDA | Random search | BERTopic |
|---|---:|---:|---:|---:|
| Topics (K) | 10 | 7 | 8 | 14 |
| Coherence (C_v) | 0.448 | 0.494 | 0.503 | **0.511** |
| Stability | 0.253 | 0.338 | 0.338 | **0.842** |
| Papers left unassigned | 0% | 0% | 0% | 17.1% |
| **Interpretability (1–5)** | 2.57 | 2.05 | 2.88 | **4.10** |

### Key findings

1. **Tuning helps.** GA-tuned LDA beat standard LDA on every measure.
2. **But random search beat the genetic algorithm** with the same budget. The gain came from
   searching the settings carefully, not from the evolutionary method itself.
3. **The best-scoring model was the hardest to interpret.** The GA settled on only 7 broad
   topics, and readers rated it lowest (2.05/5). BERTopic's topics named clear subjects and were
   rated highest (4.10/5). Good scores on an automatic measure do not guarantee topics people
   can use.
4. **What Nepal's AI research looks like:** mostly *applied* work in agriculture, landslides and
   floods, health and construction. **Generative AI in education is the fastest-rising theme**,
   and both LDA and BERTopic found this independently.

### Figures

![](paper/latex/figures/figCorpusByYear.png)
*Papers per year. Output grows sharply after 2020; 63% of the papers are from 2024–25.*

![](paper/latex/figures/figWordcloud.png)
*The most frequent words before cleaning are generic research terms, which is why they were
removed.*

![](paper/latex/figures/figCooccurrence.png)
*Word co-occurrence network of the raw text. Generic "hub" words connect everything.*

![](paper/latex/figures/figCoherenceVsK.png)
*Choosing the number of topics for standard LDA. Coherence peaks at K = 10.*

![](paper/latex/figures/figGaConvergence.png)
*The genetic algorithm's progress. It stopped improving after generation 8.*

![](paper/latex/figures/figTrends.png)
*How each theme's share changed over time. Generative AI in education rises fastest.*

---

## Task 2 — Fuzzy Logic Controller for an Assistive-Care Flat

### What we set out to do

A resident with limited mobility may not be able to reach a thermostat or light switch, so
the room has to manage heating, cooling and lighting **by itself**, in a way a carer can
understand and approve. We built this controller using **fuzzy logic**, which lets a reading
be *partly* in a category ("21 °C is somewhat cool") and makes decisions with human-style
rules. The task had three parts.

### Part 1 — Designing and building the controller

| Inputs | Outputs |
|---|---|
| Room temperature (14–34 °C) | Heating/cooling power (−100% to +100%) |
| Activity level (0–10, from motion and wrist sensors) | Lamp brightness (0–100%) |
| Daylight (0–100%) | |
| Resident's comfort preference (−5 to +5) | |

**Key design decisions**

- **Mamdani inference** rather than Sugeno, because Mamdani rules read as sentences a carer can
  check: *"IF the room is Cold AND the resident is Resting THEN heat strongly."*
- **54 rules**, generated from three small tables instead of being written one by one, so the
  whole rule base can be checked on one page.
- **Built-in safety:** activity and preference together can shift the heating decision by at
  most one step, so the controller can never heat a hot room or cool a cold one.
- **Fall-safety lighting:** a low light always stays on unless daylight is bright, because falls
  are the main injury risk for this group of residents.
- **Why activity and preference are inputs:** a resting resident produces less body heat and
  needs a warmer room, and the preference dial keeps the resident in control.
- **Technical choices:** triangular and trapezoidal membership functions; AND = minimum; rule
  outputs clipped at each rule's strength and combined with maximum; the final value taken as
  the centre of gravity (centroid).

**Worked example:** at 21 °C, activity 7.5, daylight 25% and neutral preference, 4 of the 54
rules fire. The controller commands **−22.8%** (gentle cooling) and **72.6%** lamp brightness.

**Verification:** the controller runs in both **Python and MATLAB**. MATLAB was written without
the Fuzzy Logic Toolbox, which our licence does not include, and the two versions give the
same outputs. A sweep of 70,875 input combinations found **no gaps** (every input gets an
answer) and **no unused rules**.

**Limitations found:** the centroid method can only command about 80% of full heater power,
and the heating response dips very slightly (about 0.6% of its range) at a few points.

![](paper/latex/figures/fig01_membership_functions.png)
*The fuzzy categories for all six variables.*

![](paper/latex/figures/fig15_fam_heatmaps.png)
*The complete 54-rule base shown as tables. Colour moves smoothly from cooling (blue) to
heating (red).*

![](paper/latex/figures/fig02_fuzzification_worked.png)
*Worked example, step 1: sensor readings converted into degrees of membership.*

![](paper/latex/figures/fig03_rule_activation.png)
*Worked example, step 2: 4 of the 54 rules fire.*

![](paper/latex/figures/fig04_inference_stages_hvac.png)
*Worked example, steps 3–4: rule outputs are combined into one heating/cooling command.*

![](paper/latex/figures/fig05_inference_stages_dimmer.png)
*The same steps for the lamp.*

![](paper/latex/figures/fig16_defuzzifier_comparison.png)
*Why we chose the centroid method, and its roughly 80% power limit.*

![](paper/latex/figures/fig06_surface_hvac_temp_activity.png)
*Control surface: heating/cooling against temperature and activity.*

![](paper/latex/figures/fig07_surface_hvac_temp_preference.png)
*Control surface: heating/cooling against temperature and preference.*

![](paper/latex/figures/fig08_surface_dimmer.png)
*Control surface: lamp brightness against daylight and activity.*

![](paper/latex/figures/fig09_response_hvac_by_activity.png)
*A resting resident is always given a warmer room than an active one.*

![](paper/latex/figures/fig10_operational_day.png)
*A simulated 24 hours of operation.*

![](paper/latex/figures/fig19_rule_coverage.png)
*No input goes unanswered, and every rule is used.*

![](paper/latex/figures/matlab_fig1_membership.png)
*MATLAB Online: the fuzzy categories.*

![](paper/latex/figures/matlab_fig2_ruleinference.png)
*MATLAB Online: the worked example gives the same result as Python.*

![](paper/latex/figures/matlab_fig3_surfaces.png)
*MATLAB Online: control surfaces.*

### Part 2 — Tuning the controller with a genetic algorithm

**Setup**

- A genetic algorithm adjusted the shapes of the fuzzy categories so that the controller's
  output matches a target more closely.
- **Data:** 400 examples (280 for training, 120 for testing) generated from a stated formula.
  No real dataset of residents' comfort preferences exists, so this shows the method works,
  not that real residents would be more comfortable.
- **Encoding:** 81 shape parameters × 8 bits = **a 648-bit chromosome**.
- **Genetic operators:** roulette-wheel selection, one-point crossover (70%), bit-flip mutation
  (1% per bit), and elitism (the best 2 candidates always survive). Population 60, 150
  generations, 5 runs.
- **Fitness:** 1 / (1 + error), with each output's error scaled to its range so that neither
  output dominates.

**Results**

| Method | Test error | Change vs hand design |
|---|---:|---:|
| Hand-designed controller | 0.1330 | n/a |
| Random search (same budget) | 0.1334 | 0.3% worse |
| **Genetic algorithm** | **0.0896** | **32.6% better** |

**Key findings**

- The genetic algorithm improved accuracy by **32.6%**.
- **Random search never once beat the hand design** in 45,300 tries, because the search space
  (2^648 possibilities) is far too large to search at random.
- **But tuning broke two properties it was never asked about.** In 3 of 5 runs the fuzzy
  categories fell out of order (for example, "Cold" moved above "Cool"), so the rules no longer
  meant what they said. In 4 of 5 runs, some inputs produced no command at all. **No run kept
  both properties intact.**
- **Mamdani vs Sugeno:** Sugeno would make tuning easier (400–472 bits instead of 648), but its
  rules become formulas a carer cannot check, and the safety guarantee would have to be
  enforced separately.

![](paper/latex/figures/fig11_ga_convergence.png)
*The genetic algorithm improving over 150 generations in 5 runs.*

![](paper/latex/figures/fig17_part2_control.png)
*Genetic algorithm vs random search. Random search never improved on its starting point.*

![](paper/latex/figures/fig12_mf_before_after.png)
*The fuzzy categories before and after tuning.*

![](paper/latex/figures/fig13_pred_vs_target.png)
*Controller output against the target, before and after tuning.*

![](paper/latex/figures/fig22_linguistic_ordering.png)
*Tuning put the fuzzy categories out of order in 3 of 5 runs.*

![](paper/latex/figures/fig20_error_maps.png)
*Tuning left some inputs with no command in 4 of 5 runs.*

![](paper/latex/figures/fig23_encoding_cost.png)
*Storing each parameter in 8 bits costs very little accuracy.*

### Part 3 — Comparing optimisers on standard benchmark functions

**Setup**

- Three search algorithms: a **genetic algorithm (GA)**; **particle swarm optimisation (PSO)**,
  which works like a flock of birds converging on good spots; and **simulated annealing (SA)**,
  a single searcher that gradually becomes pickier.
- Two standard **CEC'2005** test functions: **F6 Rosenbrock** (a narrow curved valley) and **F9
  Rastrigin** (thousands of small traps).
- Tested at **2 and 10 dimensions**, **15 runs** each, with an equal budget of 10,000 × D
  evaluations: 180 runs in total. We reported the mean, standard deviation, best and worst.

**Results**

| Function | 2 dimensions | 10 dimensions |
|---|---|---|
| F6 Rosenbrock | **PSO** best (solved in every run) | **GA** best on average |
| F9 Rastrigin | **PSO** best (GA effectively tied) | **GA** clearly best |
| Simulated annealing | worst in every comparison | worst in every comparison |

**Key findings**

- **No algorithm wins everywhere.** PSO is best on the easier 2-D problems; the GA is more
  reliable on the harder 10-D ones.
- **Averages can mislead.** On F6 at 10-D, a few PSO runs diverged to values in the millions,
  which inflated PSO's average. Its typical (median) run was slightly better than the GA's, and
  a statistical test found no significant difference (p = 0.62).

![](paper/latex/figures/fig21_benchmark_landscapes.png)
*The two benchmark functions, shown in 2-D.*

![](paper/latex/figures/fig14_convergence_cec2005.png)
*How quickly each algorithm improved in each of the four cases.*

![](paper/latex/figures/fig18_benchmark_distributions.png)
*Every one of the 180 runs. The spread shows what the averages hide.*

---

## Overall Conclusions

1. **Whether a genetic algorithm helps depends on the size of the search space.** In Task 1
   (3 settings), random search beat the GA. In Task 2 (648 bits), random search could not
   improve at all, while the GA gained 32.6%.
2. **Optimising a score does not guarantee what you really want.** Task 1's best-scoring model
   was the least interpretable. Task 2's most accurate controllers lost their readability and
   their full coverage of inputs.
3. **We report results honestly, including negative ones:** random search beating our proposed
   model in Task 1, and the safety cost of tuning in Task 2.

## Code and Reproducibility

All code is at **https://github.com/bpdyl/ai-research-topic-modeling**. Task 1 is a Python
pipeline with an end-to-end notebook. Task 2 has a Python implementation plus a MATLAB version
that needs no Fuzzy Logic Toolbox. All results can be regenerated from the repository.
