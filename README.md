# NYC flights 2023: predicting late and cancelled departures

A travel agency books flights from New York for its clients. About 1 flight in 4 leaves more than 15 minutes late or is cancelled, and a client who is not warned can miss a connection. This project asks: **can we tell which flights will go wrong two hours before they leave, and what should the agency do about it?**

It cleans the 2023 flights data, finds what drives delays, groups destination airports, defines the numbers to follow every month, builds two predictions, tests them honestly on months the models never saw, and turns them into two decision cards. A plain-language deck for stakeholders is in `slides/`.

## What we found

| | Result on October to December (months never used for training or tuning) |
|---|---|
| **Prediction 1: will the flight be late or cancelled?** | Warning the riskiest 2 flights in 10 reaches **40%** of the late or cancelled flights; picking at random reaches 20%. 28 warnings in 100 are right (14 flights in 100 are late). |
| **Prediction 2: how many minutes late?** | A **modest** gain: the average error is 13.12 minutes against 13.27 for a simple guess. Ranking works better: among the 5% of flights with the highest predicted delay, 21% leave 45+ minutes late, against 6% overall (3.4 times more). |
| **Weak spots** | Risk scores run too high in calm months (21% predicted, 14% real). Morning flights are reached only 6% of the time (evening: 71%). One linear model did not beat the simple guess on the test months and is reported, not used. |

The five best findings (notebook 02): delays build up through the day (13% late in the morning, 40% in the evening); a late plane makes its next flight late (61% against 25%), which matters more than the weather; airlines differ a lot (35% against 18%); cancellations come in waves on a few days; and the weather at the airports only partly explains them.

Two numbers on the decision cards are **business assumptions, not data**: the agency can warn about 2 flights in 10 and check the connections of 1 in 20. They must be confirmed with the agency. The data has no prices, so no money figure is shown.

## Folders

| Folder | What is in it |
|---|---|
| `notebooks/` | One notebook per task, run in order: `01_clean`, `02_explore`, `03_groups`, `04_metrics`, `05_features`, `06a_models_first`, `06b_models_second`, `07_decision` |
| `reports/` | Small result tables and charts that the decision cards and the slides quote |
| `slides/` | The stakeholder deck as an editable PowerPoint file (`deck.pptx`) and as `deck.pdf` exported from it, the script that builds both from `reports/`, and `source_map.csv` listing where every number on a slide comes from (also in each slide's speaker notes) |
| `src/`, `tools/`, `tests/` | Shared code, notebook tools, and the tests (including leakage and "test months untouched" checks) |
| `second-brain/` | The project's notes: decisions, results, problems, open questions, glossary, a dated log, and the owner's review checklist |
| `docs/` | The design spec and the implementation plan |

## Data

NYC flights 2023: one row per flight that left JFK, LGA or EWR, plus weather, planes, airports and airlines. Licence CC0.

Get the five files `flights.csv`, `weather.csv`, `planes.csv`, `airports.csv` and `airlines.csv` from your instructor, or from the original source https://github.com/moderndive/nycflights23, and put them in a folder called `data/` at the root of this project. **The data is never committed** (`data/` is in `.gitignore`); the notebooks write their intermediate files to `data/processed/`.

Cite: C. Ismay, S. P. Couch, H. Wickham, "nycflights23" (2025).

## How to run

```bash
pip install -r requirements.txt
python tools/run_notebooks.py          # runs the 8 notebooks in order, from a fresh kernel each (about 5 minutes)
python -m pytest tests -q              # 30 tests
python slides/build_deck.py            # rebuilds deck.pptx; the PDF is exported from it with Microsoft PowerPoint (Windows)
```

Each notebook also runs on its own with Kernel > Restart and Run All, after the earlier ones. Every random seed is fixed (`random_state=42`), so two runs give identical numbers.

## How we test honestly

- **Split by time:** models learn from January to June, are compared and tuned on July to September, then retrained on January to September. October to December is opened **once, at the very end**, in a cell marked `# ---- FINAL TEST (run once) ----`. A test checks that no cell before the marker touches those months.
- **Prediction time:** every column must be known **2 hours before** the scheduled departure. The actual departure and arrival times are never used (leakage), and the weather and the earlier flight of the same plane are only used if they were known by then. Unit tests pin this rule.
- **Dumb answers:** every model is compared with a dumb answer ("nobody is late", or "every flight leaves at the usual delay") and a simple hour-of-day rule.
- **Cut-offs by capacity:** "warn the riskiest 2 flights in 10", never a fixed probability, because the risk scores run too high in calm months.

## How AI tools were used, and what was checked by hand

**What AI did.** Claude Code (Anthropic's AI assistant) did the implementation: it wrote the code, the notebooks, the tests, the charts, the slides and the notes, working from the course task sheet and from instructions given in conversation. It also made mistakes, and recorded them in `second-brain/problems.md` together with the fixes (for example a first explanation of the cancellations that the numbers did not support, and a linear model that predicted millions of minutes).

**What the owner checked.** The owner's own review of the analysis, the numbers and the code is **in progress**, using `second-brain/review-checklist.md`. This section will list exactly the checks the owner ticked, and nothing more.

**What AI checked.** All tests pass, two full runs of the first-prediction notebook gave identical scores, and the numbers in the notebooks were compared with the report files. This is not a substitute for the owner's review.

## Second brain

`second-brain/` records every decision (`decisions.md`), result (`results.md`), problem and mistake (`problems.md`), open question, plain-language term and work session (`log.md`).
