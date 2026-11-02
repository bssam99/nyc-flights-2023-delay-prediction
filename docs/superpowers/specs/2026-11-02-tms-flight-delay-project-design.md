# TMS Flight Delay Project: Design

Date: 2026-11-02
Owner: bssam99 (with Claude Code as AI assistant)
Status: design approved in conversation; this written spec awaits the owner's review.

## 1. Purpose

Finish the eight graded tasks of the "ML week" TMS project, publish them in a new public GitHub repository, prepare a non-technical presentation for stakeholders, and keep a "second brain" folder that records every decision, result and problem.

**The business.** A travel agency that books flights from New York for its clients. The data is NYC flights 2023 (JFK, LGA, EWR), source `nycflights23`, licence CC0.

**Who it is for.**
- The instructor, who grades the eight tasks (weights in section 2).
- Stakeholders of the agency, who see the presentation and need plain language, not technical detail.
- The owner, who must be able to explain every line pushed (a course rule).

**Success criteria.** Every "Done when" line of the task sheet is met (section 2), every number on a slide can be found in the repo, every notebook runs top to bottom from the original CSVs, and the owner can explain the work from the second-brain notes.

## 2. Task sheet requirements (source: `ML_week_tasks.xlsx`)

| # | Task | Weight | File to push |
|---|---|---|---|
| 1 | Business questions, evaluate and clean data | 10% | `notebooks/01_clean.ipynb` |
| 2 | Explore with an expert eye | 10% | `notebooks/02_explore.ipynb` |
| 3 | Clustering | 10% | `notebooks/03_groups.ipynb` |
| 4 | Metrics to track | 10% | `notebooks/04_metrics.ipynb` |
| 5 | Feature engineering | 10% | `notebooks/05_features.ipynb` |
| 6 | Predict with ML and DL, test honestly | 30% | `notebooks/06a_models_first.ipynb` and `06b_models_second.ipynb` (the sheet allows one notebook per prediction) |
| 7 | Decision cards | 10% | `notebooks/07_decision.ipynb` |
| 8 | Present your work | 10% | `slides/` (PDF) |

**Rules that bind the whole project.**
- One public repository. `data/` is never committed (in `.gitignore`).
- No API keys in the repo. `.env` is ignored.
- Each notebook runs top to bottom without errors, with a plain-words markdown cell next to each step.
- Notebook outputs show only a few rows of data.
- Random seed fixed: `random_state=42`.
- The README states what the project is, the data source and citation, how to run it, and how AI tools were used and what was checked by hand.
- Citation: C. Ismay, S. P. Couch, H. Wickham, "nycflights23" (2025).
- Assumption: this is a solo project. The rules sheet is written for teams of 3 to 4, so the README and slides present it as the owner's work, assisted by Claude Code, with that disclosure stated plainly.

## 3. The data (profiled 2026-11-02, nothing modified)

| File | Rows | Notes |
|---|---|---|
| `flights.csv` | 435,352 | Main table. 14 airlines, 118 destinations, 5,117 plane tail numbers, months 1 to 12. |
| `weather.csv` | 26,207 | 3 airports. 8,736 + 8,736 + 8,735 hourly rows against 8,760 expected each. |
| `planes.csv` | 4,840 | `speed` is 0 in every row. 89 planes have no build year. |
| `airports.csv` | 1,255 | Every destination in `flights.csv` is present. |
| `airlines.csv` | 14 | |

**Train and test base rates.** "Late" means `dep_delay` above 15 minutes, or no `dep_time` (cancelled).

| | Rows | Late or cancelled | Cancelled |
|---|---|---|---|
| Train, Jan to Sep | 330,883 | 25.7% | 10,514 (3.2%) |
| Test, Oct to Dec | 104,469 | 14.1% | 224 (0.2%) |
| Validation, Jul to Sep | 108,481 | 29.1% | 5,553 |

**Consequences.**
- Test months are much calmer than training months. Fixed probability cut-offs would be off, so cut-offs are set by capacity (section 7).
- Only 224 cancellations in the test part is too few to score a cancellation model reliably, which shapes the second prediction (section 6).
- The validation part (Jul to Sep) is stormy and not like the test part, so models are chosen by ranking quality and not by a threshold tuned on it.

**Known problems for the cleaning report.**

| Problem | Rows | Planned action |
|---|---|---|
| No `dep_time` (cancelled) | 10,738 | Keep, add `cancelled` flag. They are part of the prediction target. |
| Left but no `arr_time` (diverted or similar) | 715 | Keep, add flag. Arrival columns are not used for prediction. |
| `dep_time` written as 2400 | 27 | Fix to 0000 of the next day. |
| No `tailnum` | 1,913 | Keep, add flag. Plane features stay missing. |
| Tail number not in `planes.csv` | about 2.2% of flights with a tail number | Keep, plane features missing. |
| `planes.speed` all 0 | 4,840 | Drop the column, record why. |
| `planes.year` missing | 89 | Keep, plane age missing. |
| No weather row at origin and hour | 826 flights | Fill from the latest reading within 3 hours, add flag. |
| Missing wind or visibility readings | about 1,000 to 1,200 | Leave missing for gradient boosting. Impute median plus flag for the other models. |
| Extreme delays (max 1,813 minutes) | few | Keep (real overnight delays). Report the median next to the mean. |
| Exact duplicates, impossible values (`air_time` or `distance` at or below 0) | 0 | State that they were checked and none found. |

Counts are verified again inside notebook 01, which is the source of truth for the report.

## 4. Architecture

```
nyc-flights-2023-delay-prediction/        (repo root = this folder)
├── README.md
├── .gitignore                 data/, .env, .ipynb_checkpoints, ML_week_tasks.xlsx
├── requirements.txt
├── src/common.py              paths, SEED=42, split dates, label definition, prediction-time constant
├── src/features.py            prediction-time-safe feature builders (weather lag, previous flight, congestion)
├── tools/                     py2nb.py (build notebooks), run_notebooks.py (execute in order)
├── tests/                     unit tests, hygiene tests, output-contract tests
├── notebooks/                 01_clean … 05_features, 06a_models_first, 06b_models_second, 07_decision
├── reports/                   small CSV result tables quoted by slides and cards
├── slides/                    stakeholder deck (PDF)
├── docs/superpowers/specs/    this spec
├── data/                      raw CSVs + data/processed/ (never committed)
└── second-brain/              see section 9
```

**Data flow.** Each notebook reads only the raw CSVs and files that earlier notebooks wrote, so the whole chain runs in order on a fresh clone.

| Notebook | Reads | Writes (all under `data/processed/` unless noted) |
|---|---|---|
| 01 | raw CSVs | `flights_clean.parquet`, `weather_clean.parquet`, `planes_clean.parquet`, `reports/01_data_report.csv` |
| 02 | `flights_clean` | `reports/02_findings.csv` |
| 03 | `flights_clean` (train months only to build groups) | `destination_groups.parquet`, `reports/03_groups.csv` |
| 04 | `flights_clean`, `destination_groups` | `reports/04_metrics_by_month.csv`, `reports/04_metrics_by_group.csv` |
| 05 | `flights_clean`, `weather_clean`, `planes_clean`, `destination_groups` | `features.parquet`, `reports/05_feature_table.csv` |
| 06a | `features` | `predictions_test_late.parquet`, `reports/06a_*.csv`, `reports/06_leakage_removed.csv` |
| 06b | `features`, `reports/02_second_prediction.csv` | `predictions_test_second.parquet`, `reports/06b_*.csv` |
| 07 | predictions, `destination_groups` | `reports/07_cards_numbers.csv`, `reports/07_weekly_scores.csv` |

`src/common.py` holds only what several notebooks share, so values cannot drift apart. It stays small.

## 5. The one rule behind every feature: prediction time

We predict **2 hours before the scheduled departure**, which matches the sheet's own decision example ("two hours before each flight, the agency warns clients").

**Allowed.** Anything known by then: schedule, airline, route and distance, plane details, the weather reading from 2 hours earlier, congestion in the published schedule, and the outcome of the same plane's earlier flight if it had already happened.

**Never allowed (leakage).** `dep_time`, `dep_delay`, `arr_time`, `arr_delay`, `air_time`, and anything built from them that was not yet known.

**Choices that follow.**
- *Weather*: joined on origin and the hour 2 hours before the scheduled hour, i.e. what the agency really sees at decision time. This is slightly weaker than using weather at departure and also more honest. The sheet's wording ("weather at departure") is met by joining `weather.csv` on `origin + time_hour`, with the 2-hour lag stated.
- *Previous flight of the same plane that day*: used only if its actual departure happened at least 2 hours before this flight's scheduled departure. Otherwise the feature is empty and a flag says so.
- *Destination groups* (task 3): built on training months only.

## 6. Tasks 1 to 5: analysis design

**Task 1, clean.** Three business questions (drafts, the owner may reword):
1. Which flights will leave more than 15 minutes late or be cancelled, early enough to warn clients?
2. Which airlines, airports and times of day are the most reliable, so we know what to recommend?
3. How much of the trouble is weather, and how much is a domino effect from earlier late flights?

The notebook profiles every column (type, missing, distinct values, min, max), searches for the problems in section 3, makes a fix, remove or flag decision for each with a reason, never edits the original CSVs, and ends with the data report table (problem, rows, action, reason). It also states what the data does not tell us: one year only, only flights leaving New York, no passenger data, no cause codes for delays.

**Task 2, explore.** Tables and charts for each of the three questions. At least three findings are checked with a confidence interval or a chi-square test. For each finding we ask what else could explain it. At least one pattern that looks strange is explained (for example whether "delays grow through the day" is really the domino effect). Ends with five findings, each with a chart, a number, a check and an action, and the choice of the second prediction.

**Second prediction (planned choice below, confirmed in task 2 against the stated criteria).**
- Planned: **minutes of departure delay** for flights that departed (a number). Dumb answer: the median training delay. Metric: MAE in minutes. Use: tell clients to expect a long delay and check connections.
- Rejected as the second prediction: **cancellation**. Only 224 test positives mostly from a few bad days, so the score would be noise. It remains a descriptive finding in task 2.
- Fallback: if task 2 shows minutes of delay has no learnable signal beyond the dumb answer, the second prediction becomes "more than 60 minutes late" (yes or no).

**Task 3, groups.** K-means on the 118 destination airports. Columns: average departure delay, share late, share cancelled, and delay by time of day, all computed on Jan to Sep only. Destinations with very few flights are marked and handled separately. Columns scaled. k from 2 to 10, chosen with the elbow method, the silhouette score and common sense. Each group gets a name, a size, numbers describing it, and an action. Group of each destination saved.

**Task 4, metrics.**

| Metric | Formula | Up is |
|---|---|---|
| On-time rate | flights that left at most 15 min late ÷ flights that left | good |
| Average and median departure delay | mean and median of `dep_delay`, flights that left | bad |
| Cancellation rate | flights with no `dep_time` ÷ all scheduled flights | bad |
| Share over 1 hour late | flights with `dep_delay` above 60 ÷ flights that left | bad |
| Late-or-cancelled rate | late or cancelled ÷ all scheduled flights (the first prediction target) | bad |

Computed by month and by airline, by origin airport and by destination group. Unusual months are pointed out and explained (for example the July peak, the calm October to December). Bonus: forecast Oct to Dec from Jan to Sep with a simple method and compare with what happened, noting that one year of history limits it.

**Task 5, features.** At least ten new columns, each in a table with what it is, why it could help and "known in advance: yes". Planned set: hour, weekday, month, distance, airline, origin, destination group, flights scheduled at the same origin in the same hour, plane age, seats, the six weather readings 2 hours earlier, previous flight delay and its "known" flag. Each is linked to the target with a chart or a correlation.

## 7. Tasks 6 and 7: models and decisions

**Split.** Train Jan to Sep. Fit on Jan to Jun, tune on Jul to Sep, refit on Jan to Sep with the chosen settings, then open the Oct to Dec test part once.

**First prediction.** Will the flight be late (above 15 minutes) or cancelled?
- Dumb answers: "nothing is late", and a simple hour-of-day rule (shows what the models add beyond the obvious).
- Models: logistic regression, gradient boosting, and a small PyTorch neural network. Seeded, CPU, early stopping on the validation part so no future months leak in.
- Metrics: PR-AUC and lift over the base rate for comparison; precision, recall, F1 and the confusion matrix at the cut-off. Accuracy alone is never used. A missed late flight (client stranded) costs more than a needless warning (a message).
- Also check how well risk scores match the real rates, because the test base rate is lower than the training rate.

**Second prediction.** Minutes of departure delay (section 6). Models: a linear model and gradient boosting, at least two. Compared with the median-delay answer by MAE in minutes.

**Both.** Leakage list with reasons. Permutation importance. Errors by airline, origin airport, destination group and hour band, including whether any airline or airport is treated clearly worse. Test predictions saved for task 7.

**Task 7, decision cards.** One card per prediction, the 11 lines of the sheet's example (prediction, metric and why, cut-off, action, value, model, score, weak spots, re-check when, human check, finding), all filled from code.
- *Cut-off by capacity.* For the first prediction the starting assumption is that the agency can warn about 1 flight in 5. The value is measured on the test part: how many flights are above the cut-off, how many are truly late or cancelled, and the share of all such flights caught, against picking the same number at random.
- *Second prediction.* The value is how much closer the model is than the dumb answer, in minutes, and how many truly long delays it flags.
- *Weekly scores.* The 13 test weeks are scored one by one to set a re-check trigger.
- *Groups.* A different action per destination group where it helps.
- *No invented prices.* Value is stated as flights and clients reached. If a euro figure is shown, it is labelled as an assumption the business must supply.

## 8. Task 8: the stakeholder presentation

- **Audience and tone.** Business stakeholders, not technical: plain words, one message per slide, a chart and a number on every slide, no model names without a plain translation, a glossary in the second brain.
- **Content** (from the task sheet): business questions; the data and its problems; the best findings; the destination groups; the metrics to follow; the two predictions and how they were tested honestly; the two decision cards; where AI helped and what was checked by hand. About 12 slides for a 15-minute talk.
- **Traceability.** Every number comes from a file in `reports/`.
- **Format.** Built as HTML slides and exported to PDF with Microsoft Edge (already installed). No new software.
- **Checkpoint.** The owner asked to be told before the presentation starts. After notebook 07 is pushed, the slide-by-slide outline is shown and the build waits for an explicit yes.

## 9. The second brain

Folder `second-brain/`, committed to the public repo, so it holds no secrets or personal data.

| File | Holds |
|---|---|
| `README.md` | What this folder is and how to add entries. |
| `decisions.md` | One entry per decision: id, date, decision, why, alternatives rejected. |
| `results.md` | Key numbers as they appear, each with its source notebook. |
| `problems.md` | What went wrong and how it was fixed. |
| `open-questions.md` | Things still to settle, with the task that settles them. |
| `glossary.md` | Plain-language terms used in the talk. |
| `log.md` | One dated entry per work session. |
| `done-when.md` | The task sheet's "Done when" lines as a checklist, ticked as each task finishes. |
| `deck-outline.md` | The slide-by-slide outline shown to the owner before the deck is built. |

It is updated at the end of every task, not at the end of the project. It is seeded with these decisions:

| Id | Decision |
|---|---|
| D-001 | Build one notebook at a time, in order, pushing after each. |
| D-002 | Predict 2 hours before scheduled departure. |
| D-003 | Weather joined at the hour 2 hours before the scheduled hour. |
| D-004 | Second prediction is minutes of delay, not cancellation (224 test cancellations). |
| D-005 | Cut-offs are set by capacity, not by a fixed probability. |
| D-006 | No invented prices on decision cards. |
| D-007 | The neural network is built in PyTorch, seeded, so validation can be time-ordered. |
| D-008 | `pyarrow` (intermediate files) and `pytest` (tests) are the only new dependencies. |
| D-009 | Public repository `bssam99/nyc-flights-2023-delay-prediction`; data, keys and the course spreadsheet are never committed. |
| D-010 | Show the slide outline and wait for approval before building the deck. |

## 10. GitHub and tooling

- GitHub CLI 2.103.3 is installed and signed in as `bssam99` (`repo` scope). No repository exists yet.
- After the implementation plan is approved: create the public repo `bssam99/nyc-flights-2023-delay-prediction`, write `.gitignore` first, then `git init`, first commit, push. Commit and push after each task.
- Everything needed is installed (pandas, numpy, scikit-learn, scipy, matplotlib, seaborn, torch, jupyter pieces) except `pyarrow` (intermediate files) and `pytest` (tests), which are installed when the plan starts.

## 11. Order of work

1. Repo, `.gitignore`, folder skeleton, second brain seeded, spec committed.
2. Notebooks 01 to 07, one at a time. Each is built, run from a fresh kernel, checked, committed, pushed, and logged in the second brain.
3. **Checkpoint:** slide outline shown to the owner, build waits for a yes.
4. Deck built and exported to PDF.
5. README finished (data source, citation, how to run, AI disclosure).
6. Final verification.

## 12. Verification

- Every notebook executes from a fresh kernel without errors, in order, starting from the raw CSVs only.
- The cleaned data is rebuilt from the original file with no manual steps, and the original CSVs are unchanged.
- `git status` and the repo tree show no CSV from `data/`, no `.env`, no key.
- Each task's "Done when" lines are ticked one by one in the second brain.
- Every slide number is matched to a file in `reports/`.
- The README covers every item in section 2.

## 13. Risks and how they are handled

| Risk | Handling |
|---|---|
| Test months are calmer than training | Capacity cut-offs, calibration check, week-by-week scores, stated clearly in the cards. |
| Validation months are stormier than test | Compare models by ranking quality; no threshold tuned on validation. |
| Minutes-of-delay target has heavy tails | Report MAE with the median; check the effect of the largest delays. |
| Neural network not reproducible | Fixed seeds, CPU, fixed order of data; if still unstable, report the spread over a few seeds. |
| Owner cannot explain code they did not write | Second-brain notes explain each decision in plain words; README discloses AI use honestly. |
| Public repo leaks data or keys | `.gitignore` written before `git init`; a pre-push check for CSVs and key patterns. |
