# NYC flights 2023: predicting late and cancelled departures

A travel agency books flights from New York for its clients. More than 1 flight in 5 (22.9%) leaves more than 15 minutes late or is cancelled, and a client who is not warned can miss a connection. This project asks: **can we tell which flights will go wrong two hours before they leave, and what should the agency do about it?**

It cleans the 2023 flights data, finds what drives delays, groups destination airports, defines the numbers to follow every month, builds two predictions, tests them honestly on months the models never saw, and turns them into two decision cards. A plain-language deck for stakeholders is in `slides/`.

## What we found

| | Result on October to December (kept aside: no model is trained or tuned on them; we only looked at their overall rates when designing the project) |
|---|---|
| **Prediction 1: will the flight be late or cancelled?** | Warning the riskiest 2 flights in 10 reaches **40%** of the late or cancelled flights; picking at random reaches 20%. 28 warnings in 100 are right (14 flights in 100 are late). |
| **Prediction 2: how many minutes late?** | A **modest** gain: the average error is 13.12 minutes against 13.27 for a simple guess. Ranking works better: among the 5% of flights with the highest predicted delay, 21% leave 45+ minutes late, against 6% overall (3.4 times more). |
| **Weak spots** | Risk scores run too high in calm months (22% predicted, 14% real). Morning flights are reached only 7% of the time (evening: 70%). One linear model did not beat the simple guess on the test months and is reported, not used. |

The five best findings (notebook 02): the risk of delay climbs through the day (13% late in the morning, 40% in the evening); when a plane's earlier flight ran late, its next flight is much more likely to be late (61% against 25%; 26 points of that gap remain within the same day, airport and part of the day), a bigger jump in risk than everyday bad weather brings (the rarest, most severe storms can be worse but cover very few flights), though we show a link and not a cause; airlines differ a lot (35% against 18%); cancellations come in waves on a few days; and the weather at the airports only partly explains them.

Two numbers on the decision cards are **business assumptions, not data**: the agency can warn about 2 flights in 10 and check the connections of 1 in 20. They must be confirmed with the agency. The data has no prices, so no money figure is shown.

## Folders

| Folder | What is in it |
|---|---|
| `notebooks/` | One notebook per task, run in order: `01_clean`, `02_explore`, `03_groups`, `04_metrics`, `05_features`, `06a_models_first`, `06b_models_second`, `07_decision` |
| `reports/` | Small result tables and charts that the decision cards and the slides quote |
| `slides/` | The stakeholder deck as an editable PowerPoint file (`deck.pptx`) and as `deck.pdf` exported from it, the script that builds both from `reports/`, and `source_map.csv` listing where every number on a slide comes from (also in each slide's speaker notes) |
| `src/`, `tools/`, `tests/` | Shared code, notebook tools, and the tests (including leakage and "test months untouched" checks) |

## Data

NYC flights 2023: one row per flight that left JFK, LGA or EWR, plus weather, planes, airports and airlines. Licence CC0.

Get the five files `flights.csv`, `weather.csv`, `planes.csv`, `airports.csv` and `airlines.csv` from your instructor (the dataset is the 2023 New York flights data published as `nycflights23`: https://github.com/moderndive/nycflights23), and put them in a folder called `data/` at the root of this project. Nothing depends on an exact file size: notebook 01 records a fingerprint of each file and checks that no step changes it. **The data is never committed** (`data/` is in `.gitignore`); the notebooks write their intermediate files to `data/processed/`.

Cite: C. Ismay, S. P. Couch, H. Wickham, "nycflights23" (2025).

## How to run

Developed and tested on Python 3.11.8. The pinned pandas 3 and numpy 2.4 need 3.11 or later, and the pinned PyTorch 2.5.1 has no wheels for the newest Python versions, so use Python 3.11 or 3.12.

```bash
pip install -r requirements.txt
python tools/run_notebooks.py          # runs the 8 notebooks in order, from a fresh kernel each (about 5 minutes)
python -m pytest tests -q              # 63 tests
python slides/build_deck.py            # rebuilds deck.pptx; the PDF is exported from it with Microsoft PowerPoint (Windows)
```

Each notebook also runs on its own with Kernel > Restart and Run All, after the earlier ones. Every random seed is fixed (`random_state=42`), so two runs give identical numbers.

## How we test honestly

- **Split by time:** models learn from January to June, are compared and tuned on July to September, then retrained on January to September. October to December is opened **once, at the very end**, in a cell marked `# ---- FINAL TEST (run once) ----`. A test checks that no code cell before the marker names or selects those months in the usual ways (a heuristic, not a proof). We did look at the overall late and cancelled rates of those months while designing the project (for example to choose the second prediction and the capacity rule).
- **Prediction time:** every column must be known **2 hours before** the scheduled departure. The actual departure and arrival times are never used (leakage), and the weather and the earlier flight of the same plane are only used if they were known by then. Unit tests pin this rule. An independent review found one more leak: every flight with no tail number is a cancelled flight in this data, so an empty plane would have revealed the answer. Those flights now get the plane and the earlier-flight columns of a randomly drawn real flight of the same airline and destination, and tests guard it inside United and Frontier, where almost all of them are.
- **Dumb answers:** every model is compared with a dumb answer ("nobody is late", or "every flight leaves at the usual delay") and a simple hour-of-day rule.
- **Cut-offs by capacity:** "warn the riskiest 2 flights in 10", never a fixed probability, because the risk scores run too high in calm months.

## Known limits

- **One year of data**, so seasons cannot be learned, and the October-December months were much calmer than January-September (14% late or cancelled against 26%). Risk scores run too high there; we trust the ranking, not the percentages.
- **Prediction 2 helps little on average** (1.1% closer than a simple guess on the test months); it is useful for ranking flights. One linear model did not beat the simple guess there and is reported, not used. Its choice as second prediction was a close call.
- **The destination groups are a simplification.** New random starts give the same groups (agreement 1.00), but removing the time-of-day columns (0.47) or using only January-June (0.61) changes them, and the 6-destination "thin routes" group is the least stable (4 destinations when built from January-June). Read them as useful types of situation, not natural categories.
- **Weather timestamps:** we use the reading stamped 2 hours before the scheduled hour. If readings are stamped at the end of their hour, that reading can be up to about 50 minutes newer than the decision time; a 3-hour lag would remove the doubt. The data does not say which convention applies (open question).
- **Card 1 is a list made two hours before each departure.** Warning a flight when its risk is above the cut-off set from the previous 7 days' scores reaches **40%** of the late or cancelled flights (from the 8th day of October, no outcomes or future flights used); ranking each hour's flights only among themselves (which already needs the scores of flights up to an hour ahead) would reach 34%. The headline above ranks the whole quarter together, the usual way to judge a ranking; the two agree within a point.
- **Two capacities are assumptions** to confirm with the agency (warn 2 flights in 10, check 1 in 20), and the data has no prices, so no money figure is shown.

## How AI tools were used, and what was checked by hand

**What AI did.** Claude Code (Anthropic's AI assistant) did the implementation: it wrote the code, the notebooks, the tests, the charts, the slides and the notes, working from the course task sheet and from instructions given in conversation. It also made mistakes, which were found and fixed along the way (for example a first explanation of the cancellations that the numbers did not support, and a linear model that predicted millions of minutes).

**What the owner checked.** The owner reports having reviewed all the work and completed, by themselves, checks A1 to A7 and B1 to B8 on the project as of commit `71af4be`: re-running everything from scratch, recomputing a headline number from the raw data and the value of the warning list from the saved predictions, reading how leakage is prevented, confirming that the test months stay untouched until the end, going through each of the 8 notebooks and the mistakes found along the way, and reading the leaks the independent reviews found and how they were fixed and tested. This is recorded on the owner's own word: the assistant did not do these checks and cannot verify them. Earlier the same day the owner had reported part of these checks for earlier versions; the report above is for the final one. The commits after `71af4be` only record this review, remove the assistant's notes folder from the repository (it is kept on the owner's computer) and adjust the README, the deck script and its test. The record the slides read is `slides/owner_review.json`.

**What AI checked.** All tests pass, two full runs of the first-prediction notebook gave identical scores, and the numbers in the notebooks were compared with the report files. A separate AI session then reviewed the whole project independently, without relying on these notes. It found real problems (a leak through missing tail numbers, an undisclosed change to the second-prediction probe, overstated wording about the test months and about cause and effect, and a card that could not be used as described). They were fixed or disclosed, each fix guarded by a test. A second independent review, after the nine minor fixes, found the same leak still open in a new form (planes drawn from other airlines still stood out among United's own flights), a card whose working method had not been measured, and an owner-review statement that claimed more than the history supports; these were fixed. A third independent review found that two leak channels were still open inside United and Frontier (a replacement plane that does not fit the route, and the earlier-flight columns, which are empty for every cancelled United flight), a rounding slip that made card 1 print 41% where the slides said 40%, and parts of the owner-review record that claimed more than the history supports; these were fixed in a fourth round. Where a fix changed behaviour, the test was written first and seen failing; the tests that were only strengthened passed at once because the code was already right, and one of them was also checked against a deliberately wrong input. None of this is a substitute for the owner's review.
