# TMS Flight Delay Project Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Deliver the eight graded tasks of the TMS flights project in a new public GitHub repo, with a plain-language stakeholder deck and a second-brain notes folder.

**Architecture:** Notebooks run in order. Each reads only the raw CSVs and files earlier notebooks wrote under `data/processed/`. Shared rules live in `src/`. The leakage-sensitive feature builders are unit-tested. Small CSVs in `reports/` carry every number into the decision cards and the slides.

**Tech Stack:** Python 3.11, pandas 3.0, scikit-learn 1.8, PyTorch 2.5 (CPU), scipy, matplotlib, seaborn, nbformat/nbconvert, plus new `pyarrow` and `pytest`.

**Spec:** `docs/superpowers/specs/2026-11-02-tms-flight-delay-project-design.md` (read it first; this plan references its sections as §n).

## Global Constraints

- Python 3.11 on Windows. Run every command from the repo root (this folder) with `python -m ...`. `gh` is at `C:\Program Files\GitHub CLI\gh.exe` if it is not on PATH.
- Public repo `bssam99/nyc-flights-2023-delay-prediction`. `data/` is never committed. `.env` and `ML_week_tasks.xlsx` are ignored. No API keys anywhere.
- Original CSVs are never modified (`data/flights.csv` stays 44,693,966 bytes).
- Each notebook runs top to bottom without errors from the raw CSVs plus earlier notebooks' outputs only. Each step has a plain-words markdown cell next to its code. Outputs show only a few rows.
- Seed: `random_state=42` everywhere (`common.SEED`).
- Late = `dep_delay` above 15 minutes, or no `dep_time`.
- Train = Jan to Sep, test = Oct to Dec. Fit = Jan to Jun, validation = Jul to Sep. The test part is opened once per prediction, at the end.
- Prediction time = 2 hours before the scheduled departure. Never use `dep_time`, `dep_delay`, `arr_time`, `arr_delay`, `air_time`, or anything built from them that was not yet known, as a feature.
- Cut-offs are set by capacity. No invented prices: a euro figure only if labelled as an assumption.
- Cite the data: C. Ismay, S. P. Couch, H. Wickham, "nycflights23" (2025), CC0.
- The README states what AI was used for and what the owner checked. It must not claim a check the owner did not do.
- After each task: update the second brain, commit, push. Every commit message ends with the line `Co-Authored-By: Claude Sonnet 5.5 <noreply@anthropic.com>`.
- The deck is non-technical, about 12 slides, exported to PDF, built only after the owner approves the outline (Task 11). Every number comes from a file in `reports/`.

## Review Focus

1. **Prediction-time leakage in the previous-flight and weather features** (a feature that uses a departure not yet happened at prediction time). Pinned by exact-case tests in Task 7.
2. **Joins that silently drop flights** with no tail number, no plane row or no weather row. Row counts are asserted in Tasks 3 and 7.
3. **The test months used before the final evaluation.** A guard test on the model notebooks in Tasks 8 and 9.
4. **A fresh clone with no `data/` folder, or an earlier notebook not yet run.** The error must say what is missing and where to get it. Pinned in Task 2.
5. **Rare destinations** (few flights) forming unstable clusters or NaN inputs to k-means. Pinned in Task 5.

## File Structure

| Path | Responsibility |
|---|---|
| `src/common.py` | Paths, seed, constants, label definition, split into fit/valid/test, raw-file check |
| `src/features.py` | Prediction-time-safe feature builders (congestion, lagged weather, previous flight) |
| `tools/py2nb.py` | Convert a `# %%` percent-format script into a `.ipynb` |
| `tools/run_notebooks.py` | Execute notebooks in order, in place, from a fresh kernel |
| `tests/test_repo_hygiene.py` | No data, keys or spreadsheets tracked; `.gitignore` entries present |
| `tests/test_common.py`, `test_py2nb.py`, `test_run_notebooks.py`, `test_features.py` | Unit tests |
| `tests/test_notebooks_hygiene.py` | Executed-notebook rules (no errors, small outputs, no local paths, test opened once) |
| `tests/test_outputs.py` | One output-contract test per notebook ("Done when" lines), plus the deck |
| `notebooks/` | `01_clean`, `02_explore`, `03_groups`, `04_metrics`, `05_features`, `06a_models_first`, `06b_models_second`, `07_decision` |
| `reports/` | Small CSVs and `figures/` PNGs quoted by cards and slides |
| `slides/` | `build_deck.py`, `deck.html`, `deck.pdf`, `source_map.csv`, `assets/` |
| `second-brain/` | Notes (spec §9) plus `done-when.md` and `deck-outline.md` |

Notebook source scripts (`# %%` format) are written in a scratch directory outside the repo and are not committed; the executed `.ipynb` is the source of truth after the build. Every notebook starts with this setup cell (each notebook after 01 then calls `common.require_processed(...)` with the files it reads):

```python
import sys
from pathlib import Path
sys.path.insert(0, str(Path.cwd().parent))
from src import common
```

---

### Task 1: Repo bootstrap and safety net

**Files:**
- Create: `.gitignore`, `requirements.txt`, `README.md` (short stub), `tests/test_repo_hygiene.py`, `second-brain/README.md`, `decisions.md`, `results.md`, `problems.md`, `open-questions.md`, `glossary.md`, `log.md`, `done-when.md`

**Interfaces:**
- Produces: public GitHub repo with `origin/main`; second-brain entry formats `D-NNN` (decision), `R-NNN` (result), `P-NNN` (problem), `## YYYY-MM-DD` (log); `done-when.md` holding every "Done when" line of the task sheet as `- [ ]` boxes, grouped by task.

- [ ] **Step 1: Install the two new dependencies**

Run: `python -m pip install pyarrow pytest` then `python -c "import pyarrow, pytest; print(pyarrow.__version__, pytest.__version__)"`
Expected: both versions print.

- [ ] **Step 2: Write `.gitignore`** with these entries: `data/`, `.env`, `.ipynb_checkpoints/`, `__pycache__/`, `.pytest_cache/`, `ML_week_tasks.xlsx`.

- [ ] **Step 3: Write the failing hygiene tests** in `tests/test_repo_hygiene.py`

```python
import re, subprocess
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SECRET = re.compile(r"(sk-[A-Za-z0-9]{20,}|gh[pousr]_[A-Za-z0-9]{30,}|AIza[0-9A-Za-z_\-]{30,})")

def _tracked():
    out = subprocess.run(["git", "ls-files"], cwd=ROOT, capture_output=True, text=True, check=True).stdout
    return out.split("\n")[:-1]

def test_no_data_keys_or_spreadsheets_tracked():
    bad = [f for f in _tracked() if f.startswith("data/") or f == ".env" or f.endswith(".xlsx")
           or (f.endswith(".csv") and not f.startswith("reports/") and not f.startswith("slides/"))]
    assert bad == []

def test_gitignore_has_required_entries():
    lines = (ROOT / ".gitignore").read_text().split("\n")
    for entry in ["data/", ".env", "ML_week_tasks.xlsx"]:
        assert entry in lines

def test_no_secret_patterns_in_tracked_text_files():
    hits = [f for f in _tracked() if f.endswith((".py", ".md", ".txt", ".ipynb", ".html", ".csv"))
            and SECRET.search((ROOT / f).read_text(errors="ignore"))]
    assert hits == []
```

- [ ] **Step 4: Run to see it fail**

Run: `python -m pytest tests/test_repo_hygiene.py -v`
Expected: FAIL with `fatal: not a git repository` (or CalledProcessError).

- [ ] **Step 5: Initialise git and check nothing forbidden is staged**

Run: `git init -b main`, then `git add .gitignore`, then `git status --short`
Expected: lists `.gitignore`; `data/` and `ML_week_tasks.xlsx` do not appear.

- [ ] **Step 6: Write `requirements.txt`** (exact installed versions, from `python -m pip freeze`, of: pandas, numpy, scikit-learn, scipy, matplotlib, seaborn, torch, pyarrow, nbformat, nbconvert, ipykernel, pytest). Write the second-brain files:
  - `README.md`: purpose, the four entry formats above, and the rule "update at the end of every task".
  - `decisions.md`: entries D-001 to D-010 copied from spec §9, each with date 2026-11-02, decision, why, alternatives rejected. Add D-011: two model notebooks (`06a`, `06b`), because the sheet allows one per prediction and it makes the test-opened-once guard enforceable.
  - `results.md`, `problems.md`, `open-questions.md`, `glossary.md`, `log.md`: headed and empty, `log.md` with a first entry for today.
  - `done-when.md`: unticked checklist of all "Done when" lines for tasks 1 to 8.
  - Repo `README.md` stub: title, one-line purpose, the data citation, "work in progress, see `second-brain/`".

- [ ] **Step 7: Stage everything, run the tests**

Run: `git add -A` then `python -m pytest tests/test_repo_hygiene.py -v`
Expected: 3 PASS. Also stage the spec and this plan (`docs/`).

- [ ] **Step 8: Commit, create the public repo, push**

```bash
git commit -m "chore: bootstrap repo, second brain, spec and plan"
gh repo create nyc-flights-2023-delay-prediction --public --source . --remote origin --push \
  --description "NYC flights 2023: predicting late and cancelled departures for a travel agency"
gh repo view bssam99/nyc-flights-2023-delay-prediction --json visibility,url
```
Expected: `"visibility":"PUBLIC"`, and the GitHub file list shows no `data/` and no `.xlsx`.

---

### Task 2: Shared module and notebook tooling

**Files:**
- Create: `src/__init__.py`, `src/common.py`, `tools/py2nb.py`, `tools/run_notebooks.py`, `tests/test_common.py`, `tests/test_py2nb.py`, `tests/test_run_notebooks.py`, `tests/test_notebooks_hygiene.py`

**Interfaces:**
- Produces (`src/common.py`):
  - constants `SEED=42`, `LEAD_HOURS=2`, `LATE_MINUTES=15`, `FLIGHT_ROWS=435_352`; paths `ROOT`, `RAW` (= `ROOT/"data"`), `PROC` (= `RAW/"processed"`), `REPORTS` (= `ROOT/"reports"`)
  - `require_raw_files() -> None`: raises `FileNotFoundError` naming each missing CSV (`flights, weather, planes, airports, airlines`) and the README section "Data"
  - `make_late_label(df: pd.DataFrame) -> pd.Series`: int8, 1 if `dep_time` is NaN or `dep_delay` > `LATE_MINUTES`
  - `split_part(month: pd.Series) -> pd.Series`: `"fit"` for 1 to 6, `"valid"` for 7 to 9, `"test"` for 10 to 12
  - `require_processed(*names: str) -> None`: raises `FileNotFoundError` saying which earlier notebook writes each missing `PROC` file
- Produces (`tools/py2nb.py`): `convert(source: str) -> nbformat.NotebookNode`; CLI `python tools/py2nb.py in.py out.ipynb`. A line `# %% [markdown]` starts a markdown cell (lines stripped of a leading `# `); `# %%` starts a code cell.
- Produces (`tools/run_notebooks.py`): `run_notebook(path: Path, timeout: int = 3600) -> None` executes in place with kernel `python3`, cwd = the notebook's folder, raises on any cell error; CLI `python tools/run_notebooks.py [01 02 ...]` runs the notebooks whose filename starts with each prefix, in order (no args = all), exit code 1 on failure.

- [ ] **Step 1: Write the failing tests**

```python
# tests/test_common.py
import pandas as pd, pytest
from src import common

def test_make_late_label():
    df = pd.DataFrame({"dep_time": [None, 500, 500, 500], "dep_delay": [None, 15.0, 16.0, -5.0]})
    assert common.make_late_label(df).tolist() == [1, 0, 1, 0]

def test_split_part_maps_months():
    parts = common.split_part(pd.Series(range(1, 13))).tolist()
    assert parts == ["fit"] * 6 + ["valid"] * 3 + ["test"] * 3

def test_require_raw_files_names_missing_file_and_readme(tmp_path, monkeypatch):
    monkeypatch.setattr(common, "RAW", tmp_path)
    with pytest.raises(FileNotFoundError, match=r"flights\.csv.*README"):
        common.require_raw_files()

def test_require_processed_names_the_notebook(tmp_path, monkeypatch):
    monkeypatch.setattr(common, "PROC", tmp_path)
    with pytest.raises(FileNotFoundError, match="01_clean"):
        common.require_processed("flights_clean.parquet")
```
```python
# tests/test_py2nb.py
from tools.py2nb import convert

def test_convert_splits_markdown_and_code():
    nb = convert("# %% [markdown]\n# Hello\n# %%\nx = 1\n")
    assert [c.cell_type for c in nb.cells] == ["markdown", "code"]
    assert nb.cells[0].source == "Hello" and nb.cells[1].source == "x = 1"
```
```python
# tests/test_run_notebooks.py
import nbformat, pytest
from tools.run_notebooks import run_notebook

def _nb(path, code):
    nb = nbformat.v4.new_notebook(cells=[nbformat.v4.new_code_cell(code)])
    nbformat.write(nb, path)

def test_run_notebook_stores_output(tmp_path):
    p = tmp_path / "a.ipynb"; _nb(p, "print('hi')")
    run_notebook(p)
    assert "hi" in nbformat.read(p, 4).cells[0].outputs[0].text

def test_run_notebook_raises_on_error(tmp_path):
    p = tmp_path / "b.ipynb"; _nb(p, "1/0")
    with pytest.raises(Exception):
        run_notebook(p)
```
`tests/test_notebooks_hygiene.py` is parametrised over `notebooks/*.ipynb` (passes vacuously while none exist) and asserts for each: no output of type `error`; every code cell has `execution_count`; no cell's text output exceeds 6,000 characters; markdown cells number at least half the code cells; the first code cell contains `from src import common`; the raw notebook JSON contains neither `C:\\Users` nor `C:/Users`.

- [ ] **Step 2: Run to see them fail**

Run: `python -m pytest tests -v`
Expected: FAIL with `ModuleNotFoundError: src` / `tools`.

- [ ] **Step 3: Implement `src/common.py`, `tools/py2nb.py`, `tools/run_notebooks.py`** to the interfaces above. `run_notebook` uses `nbconvert.preprocessors.ExecutePreprocessor` (not the CLI). `require_processed` maps file name to notebook with a dict (e.g. `flights_clean.parquet` -> `01_clean`).

- [ ] **Step 4: Run the tests**

Run: `python -m pytest tests -v`
Expected: all PASS.

- [ ] **Step 5: Second brain and commit.** Add a `log.md` entry; add decision D-012 "notebooks are built from scratch `# %%` scripts; the executed `.ipynb` is the source of truth; no jupytext". Then:

```bash
git add -A && git commit -m "feat: shared module and notebook tooling" && git push
```

---

### Task 3: Notebook 01, business questions and clean data

**Files:**
- Create: `notebooks/01_clean.ipynb`
- Modify: `tests/test_outputs.py` (create it), `second-brain/*`

**Interfaces:**
- Consumes: raw CSVs, `common.require_raw_files`, `common.make_late_label`, `common.split_part`.
- Produces under `data/processed/`:
  - `flights_clean.parquet`: all 435,352 rows, original columns (`time_hour` as datetime, `dep_time` 2400 -> 0) plus `row_id` (0 to 435,351, original order), `sched_dep_dt` (= `time_hour` + `minute` minutes), `cancelled`, `diverted`, `no_tailnum` (int8), `late_or_cancelled` (int8), `part` (`fit`/`valid`/`test`)
  - `weather_clean.parquet`: full hourly grid, 3 airports x 8,760 = 26,280 rows, gaps forward-filled up to 3 hours within each airport, plus `weather_filled` (int8)
  - `planes_clean.parquet`: 4,840 rows, `speed` dropped, plus `year_missing`
- Produces `reports/01_data_report.csv`: columns `problem, rows, action, reason`.

- [ ] **Step 1: Write the failing contract test** in `tests/test_outputs.py`

```python
import pandas as pd
from src import common

def _csv(name): return pd.read_csv(common.REPORTS / name)

def test_task1_clean_data_and_report():
    r = _csv("01_data_report.csv")
    assert list(r.columns) == ["problem", "rows", "action", "reason"]
    assert r["reason"].str.len().gt(10).all() and (r["rows"] >= 0).all()
    rows = dict(zip(r["problem"], r["rows"]))
    assert rows["cancelled"] == 10_738 and rows["diverted"] == 715 and rows["dep_time_2400"] == 27
    assert rows["missing_tailnum"] == 1_913 and rows["planes_speed_all_zero"] == 4_840
    assert rows["planes_year_missing"] == 89 and rows["weather_gap"] == 826
    assert {"tailnum_not_in_planes", "weather_values_missing", "extreme_delay",
            "duplicates_and_impossible_values"} <= set(rows)
    f = pd.read_parquet(common.PROC / "flights_clean.parquet")
    assert len(f) == common.FLIGHT_ROWS and f["row_id"].is_unique and f["dep_time"].max() < 2400
    assert len(pd.read_parquet(common.PROC / "weather_clean.parquet")) == 26_280
    assert (common.RAW / "flights.csv").stat().st_size == 44_693_966
```

- [ ] **Step 2: Run to see it fail**

Run: `python -m pytest tests/test_outputs.py::test_task1_clean_data_and_report -v`
Expected: FAIL (file not found).

- [ ] **Step 3: Build `01_clean`** per spec §3 and §6 Task 1:
  - The three business questions at the top, as in §6 (the owner may have reworded them).
  - A profile table per file: column, dtype, missing, distinct values, min, max.
  - One check per problem listed in §3; the `problem` names in the report are exactly: `cancelled, diverted, dep_time_2400, missing_tailnum, tailnum_not_in_planes, planes_speed_all_zero, planes_year_missing, weather_gap, weather_values_missing, extreme_delay, duplicates_and_impossible_values`.
  - `weather_gap` = flights with no weather row at their own `(origin, time_hour)`.
  - A "what the data does not tell us" cell (one year, only New York departures, no passengers, no delay causes).
  - Cleaned files saved with `to_parquet`; the final cell displays the report.

- [ ] **Step 4: Convert and execute**

Run: `python tools/py2nb.py <scratch>/01_clean.py notebooks/01_clean.ipynb` then `python tools/run_notebooks.py 01`
Expected: exit 0.

- [ ] **Step 5: Run the tests**

Run: `python -m pytest tests -v`
Expected: all PASS (contract test and hygiene tests).

- [ ] **Step 6: Second brain.** One `P-NNN` per report row in `problems.md`; key counts in `results.md`; `log.md`; tick Task 1 in `done-when.md`.

- [ ] **Step 7: Commit and push:** `git add -A && git commit -m "feat: notebook 01 business questions and data cleaning" && git push`

---

### Task 4: Notebook 02, explore and choose the second prediction

**Files:**
- Create: `notebooks/02_explore.ipynb`
- Modify: `tests/test_outputs.py`, `second-brain/*`

**Interfaces:**
- Consumes: `flights_clean.parquet`.
- Produces: `reports/02_findings.csv` (columns `rank, finding, number, check_type, check_result, action, chart_file, looks_strange_at_first, why`, exactly 5 rows); `reports/figures/02_*.png` (one per finding); `reports/02_second_prediction.csv` (columns `choice, target, metric, dumb_answer, reason`, one row, `choice` is `minutes_late` or `over_60_min`).

- [ ] **Step 1: Write the failing contract test**

```python
def test_task2_findings_and_second_prediction():
    f = _csv("02_findings.csv")
    assert len(f) == 5 and f["action"].str.len().gt(10).all()
    assert (f["check_type"] != "none").sum() >= 3
    assert set(f["check_type"]) <= {"confidence_interval", "chi_square", "none"}
    assert all((common.REPORTS / "figures" / p).exists() for p in f["chart_file"])
    strange = f[f["looks_strange_at_first"]]
    assert len(strange) >= 1 and strange["why"].str.len().gt(20).all()
    s = _csv("02_second_prediction.csv")
    assert len(s) == 1 and s.loc[0, "choice"] in {"minutes_late", "over_60_min"}
```

- [ ] **Step 2: Run to see it fail.** `python -m pytest tests/test_outputs.py::test_task2_findings_and_second_prediction -v` -> FAIL.

- [ ] **Step 3: Build `02_explore`** per spec §6 Task 2:
  - Tables and charts for each of the three questions (when, where, who, how much).
  - At least three findings checked: Wilson or bootstrap confidence interval for a rate, or `scipy.stats.chi2_contingency` for two percentages.
  - For each finding a "what else could explain it" test (e.g. the domino effect for "delays grow through the day").
  - Second-prediction probe, with data from the fit/valid parts only: a `HistGradientBoostingRegressor` on schedule columns (hour, weekday, month, carrier, origin, distance), compared with the median-delay answer by MAE on the valid part. Rule (decision D-013): choose `minutes_late` unless the model's valid MAE is not at least 3% lower than the median answer's; then choose `over_60_min`. Cancellation stays a descriptive finding only (spec §6).
  - Save charts to `reports/figures/`.

- [ ] **Step 4: Convert and execute** (`02`), expect exit 0.
- [ ] **Step 5: Run the tests,** expect all PASS.
- [ ] **Step 6: Second brain.** D-013 (second-prediction rule and result), five findings in `results.md`, `log.md`, tick Task 2.
- [ ] **Step 7: Commit and push:** `git add -A && git commit -m "feat: notebook 02 exploration and second prediction" && git push`

---

### Task 5: Notebook 03, destination groups

**Files:**
- Create: `notebooks/03_groups.ipynb`
- Modify: `tests/test_outputs.py`, `second-brain/*`

**Interfaces:**
- Consumes: `flights_clean.parquet`.
- Produces: `data/processed/destination_groups.parquet` (columns `dest, group_id, group_name`; one row per each of the 118 destinations; `group_id = -1`, name `Rare destinations` for destinations with fewer than 100 flights in Jan to Sep); `reports/03_groups.csv` (columns `group_name, size, typical, action`); `reports/03_k_selection.csv` (columns `k, inertia, silhouette`, k = 2 to 10).

- [ ] **Step 1: Write the failing contract test**

```python
def test_task3_groups():
    g = pd.read_parquet(common.PROC / "destination_groups.parquet")
    assert g["dest"].is_unique and len(g) == 118 and g["group_name"].notna().all()
    r = _csv("03_groups.csv")
    assert r[["group_name", "typical", "action"]].notna().all().all()
    assert r["size"].sum() == 118
    assert list(_csv("03_k_selection.csv")["k"]) == list(range(2, 11))
```

- [ ] **Step 2: Run to see it fail.** Expected FAIL (file not found).
- [ ] **Step 3: Build `03_groups`** per spec §6 Task 3, using only `part != "test"` flights. Per-destination columns, exactly: `avg_dep_delay, share_late, share_cancelled, delay_night, delay_morning, delay_afternoon, delay_evening` (hour bands 0 to 5, 6 to 11, 12 to 17, 18 to 23; a missing band takes the destination's overall average delay, so no NaN reaches k-means, asserted in the notebook). `StandardScaler`, then `KMeans(n_init=10, random_state=common.SEED)` for k = 2 to 10; pick k with silhouette and common sense and say why in a markdown cell. Name each group, describe it with numbers, give one action each. The `Rare destinations` set is excluded from clustering and appears as its own row.
- [ ] **Step 4: Convert and execute** (`03`), expect exit 0.
- [ ] **Step 5: Run the tests,** expect all PASS.
- [ ] **Step 6: Second brain** (group names and sizes in `results.md`, k decision in `decisions.md`, `log.md`, tick Task 3).
- [ ] **Step 7: Commit and push:** `git add -A && git commit -m "feat: notebook 03 destination groups" && git push`

---

### Task 6: Notebook 04, metrics

**Files:**
- Create: `notebooks/04_metrics.ipynb`
- Modify: `tests/test_outputs.py`, `second-brain/*`

**Interfaces:**
- Consumes: `flights_clean.parquet`, `destination_groups.parquet`.
- Produces: `reports/04_metric_definitions.csv` (`metric, formula, why, up_is`); `reports/04_metrics_by_month.csv` (columns `month, flights, on_time_rate, avg_dep_delay, median_dep_delay, cancellation_rate, share_over_60, late_or_cancelled_rate`, 12 rows); `reports/04_metrics_by_group.csv` (same metric columns plus `dimension, level`, for carrier, origin and destination group); `reports/04_forecast_check.csv` (`month, actual, forecast, error` for months 10 to 12, bonus).

- [ ] **Step 1: Write the failing contract test**

```python
def test_task4_metrics():
    d = _csv("04_metric_definitions.csv")
    assert 4 <= len(d) <= 6 and d[["formula", "why", "up_is"]].notna().all().all()
    m = _csv("04_metrics_by_month.csv").set_index("month")
    assert len(m) == 12 and m["on_time_rate"].between(0, 1).all()
    assert abs(m.loc[7, "late_or_cancelled_rate"] - 0.391) < 0.001
    assert abs(m.loc[7, "cancellation_rate"] - 0.087) < 0.001
    assert abs(m.loc[10, "late_or_cancelled_rate"] - 0.132) < 0.001
    g = _csv("04_metrics_by_group.csv")
    assert {"carrier", "origin", "dest_group"} <= set(g["dimension"])
```

- [ ] **Step 2: Run to see it fail.** Expected FAIL.
- [ ] **Step 3: Build `04_metrics`** with the five metrics and formulas of spec §6 Task 4 written in markdown cells before the code. Departed-only denominators for the delay metrics; all scheduled flights for cancellation and late-or-cancelled. Point out unusual months with a reason. Bonus: forecast months 10 to 12 from months 1 to 9 with a simple stated method and compare with actuals, noting the one-year limit.
- [ ] **Step 4: Convert and execute** (`04`), expect exit 0.
- [ ] **Step 5: Run the tests,** expect all PASS.
- [ ] **Step 6: Second brain** (metric table to `results.md`, `log.md`, tick Task 4).
- [ ] **Step 7: Commit and push:** `git add -A && git commit -m "feat: notebook 04 metrics" && git push`

---

### Task 7: Feature builders and notebook 05

**Files:**
- Create: `src/features.py`, `tests/test_features.py`, `notebooks/05_features.ipynb`
- Modify: `tests/test_outputs.py`, `second-brain/*`

**Interfaces:**
- Consumes: `flights_clean` (needs `row_id, tailnum, sched_dep_dt, dep_delay, time_hour, origin`), `weather_clean`, `planes_clean`, `destination_groups`.
- Produces (`src/features.py`), each returns a new DataFrame with the same rows in `row_id` order:
  - `add_congestion(f) -> DataFrame`: adds `origin_hour_flights` (int): scheduled departures sharing `origin` and `time_hour`
  - `add_lagged_weather(f, w, lead_hours=common.LEAD_HOURS) -> DataFrame`: left join on `origin` and `time_hour - lead_hours`; adds `temp, wind_speed, wind_gust, visib, precip, pressure, weather_filled`
  - `add_previous_flight(f, lead_hours=common.LEAD_HOURS) -> DataFrame`: adds `prev_dep_delay` (float, NaN unless known) and `prev_known` (int8). The previous flight is the same `tailnum`'s closest earlier flight on the same calendar date; it is known only if `sched_dep_dt + dep_delay` of that flight is at or before this flight's `sched_dep_dt - lead_hours`, and its `dep_delay` is not NaN.
- Produces: `data/processed/features.parquet` (all 435,352 rows: `row_id, part, late_or_cancelled, dep_delay` as targets, plus every feature column); `reports/05_feature_table.csv` (`feature, what, why, known_in_advance`).

- [ ] **Step 1: Write the failing unit tests** in `tests/test_features.py`

```python
import numpy as np, pandas as pd
from src.features import add_congestion, add_lagged_weather, add_previous_flight

def test_previous_flight_known_only_if_left_before_prediction_time():
    rows = [(0, "N1", "2023-05-01 06:00", 30.0), (1, "N1", "2023-05-01 08:00", 60.0),
            (2, "N1", "2023-05-01 09:30", 0.0),  (3, "N1", "2023-05-01 12:00", 10.0),
            (4, "N2", "2023-05-01 12:00", 99.0), (5, "N1", "2023-05-02 06:00", -5.0),
            (6, "N3", "2023-05-01 08:00", np.nan), (7, "N3", "2023-05-01 12:00", 1.0)]
    f = pd.DataFrame(rows, columns=["row_id", "tailnum", "sched_dep_dt", "dep_delay"])
    f["sched_dep_dt"] = pd.to_datetime(f["sched_dep_dt"])
    out = add_previous_flight(f).set_index("row_id")
    assert out["prev_known"].tolist() == [0, 0, 0, 1, 0, 0, 0, 0]
    assert out.loc[3, "prev_dep_delay"] == 0.0      # prev left 09:30, decision time 10:00
    assert out.loc[[0, 1, 2, 4, 5, 6, 7], "prev_dep_delay"].isna().all()

def test_lagged_weather_uses_reading_two_hours_earlier_and_keeps_rows():
    f = pd.DataFrame({"row_id": [0, 1], "origin": ["JFK", "JFK"],
                      "time_hour": pd.to_datetime(["2023-05-01 10:00", "2023-05-01 03:00"])})
    hours = pd.date_range("2023-05-01 07:00", periods=4, freq="h")
    w = pd.DataFrame({"origin": "JFK", "time_hour": hours, "temp": [1., 2., 3., 4.],
                      "wind_speed": 0., "wind_gust": 0., "visib": 10., "precip": 0.,
                      "pressure": 1000., "weather_filled": 0})
    out = add_lagged_weather(f, w)
    assert len(out) == 2 and out.loc[0, "temp"] == 2.0 and np.isnan(out.loc[1, "temp"])

def test_congestion_counts_flights_sharing_origin_and_hour():
    f = pd.DataFrame({"row_id": range(4), "origin": ["JFK"] * 3 + ["EWR"],
                      "time_hour": pd.to_datetime(["2023-05-01 10:00"] * 4)})
    assert add_congestion(f)["origin_hour_flights"].tolist() == [3, 3, 3, 1]
```

- [ ] **Step 2: Run to see them fail.** `python -m pytest tests/test_features.py -v` -> FAIL (`ModuleNotFoundError: src.features`).
- [ ] **Step 3: Implement `src/features.py`** to the interfaces. `add_previous_flight` sorts by `tailnum, sched_dep_dt`, uses `groupby(...).shift` and the same-calendar-date rule, and returns rows in `row_id` order.
- [ ] **Step 4: Run to see them pass.** `python -m pytest tests/test_features.py -v` -> 3 PASS.
- [ ] **Step 5: Write the failing contract test**

```python
FORBIDDEN = {"dep_time", "dep_delay", "arr_time", "arr_delay", "air_time"}

def test_task5_features():
    t = _csv("05_feature_table.csv")
    assert len(t) >= 10 and (t["known_in_advance"] == "yes").all()
    assert t[["what", "why"]].notna().all().all() and not (set(t["feature"]) & FORBIDDEN)
    f = pd.read_parquet(common.PROC / "features.parquet")
    assert len(f) == common.FLIGHT_ROWS and set(t["feature"]) <= set(f.columns)
    assert {"row_id", "part", "late_or_cancelled", "dep_delay"} <= set(f.columns)
```

- [ ] **Step 6: Run to see it fail.** Expected FAIL (file not found).
- [ ] **Step 7: Build `05_features`** per spec §5 and §6 Task 5. Planned feature set: `hour, weekday, month, distance, carrier, origin, dest_group, origin_hour_flights, plane_age, seats, temp, wind_speed, wind_gust, visib, precip, pressure, prev_dep_delay, prev_known, weather_filled, plane_known`. `plane_age = 2023 - year` (NaN if unknown). `dest_group` is `group_id` from task 5. The notebook asserts the row count is unchanged after every join, links each feature to the targets with a chart or correlation, and states the 2-hour weather and previous-flight rules in markdown.
- [ ] **Step 8: Convert and execute** (`05`), expect exit 0.
- [ ] **Step 9: Run all tests,** expect PASS.
- [ ] **Step 10: Second brain** (feature table summary to `results.md`, D-003 confirmed with the weather rule, `log.md`, tick Task 5).
- [ ] **Step 11: Commit and push:** `git add -A && git commit -m "feat: feature builders and notebook 05" && git push`

---

### Task 8: Notebook 06a, first prediction (late or cancelled)

**Files:**
- Create: `notebooks/06a_models_first.ipynb`
- Modify: `tests/test_outputs.py`, `tests/test_notebooks_hygiene.py`, `second-brain/*`

**Interfaces:**
- Consumes: `features.parquet`, `destination_groups.parquet`.
- Produces: `data/processed/predictions_test_late.parquet` (`row_id, risk, y_true`; exactly the 104,469 test rows); `reports/06a_valid_results.csv` and `reports/06a_results.csv` (columns `model, metric, value`; models `dumb_never_late, hour_rule, logistic_regression, gradient_boosting, neural_network`; metrics `pr_auc, precision_at_20pct, recall_at_20pct, f1_at_20pct`); `reports/06a_confusion.csv` (best model at the top-20% cut-off); `reports/06a_calibration.csv` (risk deciles: mean predicted, actual rate, n); `reports/06_leakage_removed.csv` (`column, reason`); `reports/06a_importance.csv` (`feature, importance_mean, importance_std`); `reports/06a_group_errors.csv` (`dimension, level, n, metric, value` for carrier, origin, dest_group, hour_band).

- [ ] **Step 1: Write the failing tests**

```python
def test_task6a_first_prediction():
    r = _csv("06a_results.csv")
    pr = r[r["metric"] == "pr_auc"].set_index("model")["value"]
    assert {"dumb_never_late", "hour_rule", "logistic_regression", "gradient_boosting",
            "neural_network"} <= set(pr.index)
    for m in ["logistic_regression", "gradient_boosting", "neural_network"]:
        assert pr[m] > pr["dumb_never_late"]
    p = pd.read_parquet(common.PROC / "predictions_test_late.parquet")
    f = pd.read_parquet(common.PROC / "features.parquet", columns=["row_id", "part"])
    assert set(p["row_id"]) == set(f.loc[f["part"] == "test", "row_id"]) and p["risk"].between(0, 1).all()
    assert {"dep_time", "dep_delay", "arr_time", "arr_delay", "air_time"} <= set(_csv("06_leakage_removed.csv")["column"])
    assert {"carrier", "origin", "dest_group", "hour_band"} <= set(_csv("06a_group_errors.csv")["dimension"])
```
In `tests/test_notebooks_hygiene.py` add `test_test_part_opened_once`, parametrised over `06a` and `06b` notebooks: find the first code cell containing the marker `# ---- FINAL TEST (run once) ----`; assert no code cell before it contains the string `'test'` or `"test"`.

- [ ] **Step 2: Run to see them fail.** Expected FAIL (file not found).
- [ ] **Step 3: Build `06a_models_first`** per spec §7. Pinned choices:
  - Sections: leakage list, dumb answers (`dumb_never_late` = constant score, so PR-AUC equals the base rate; `hour_rule` = training late rate by scheduled hour), tuning on valid, refit on Jan to Sep, final test.
  - Logistic regression: one-hot for `carrier, origin, dest_group`, median imputation, standardised numerics; `C` from `[0.01, 0.1, 1, 10]`.
  - `HistGradientBoostingClassifier` with native categoricals, `random_state=common.SEED`; grid `learning_rate` in `[0.05, 0.1]` x `max_leaf_nodes` in `[15, 31, 63]`.
  - Neural network in PyTorch on the logistic preprocessing: hidden layers 64 and 32 (ReLU, dropout 0.2), `BCEWithLogitsLoss`, Adam lr 1e-3, batch 2048, up to 30 epochs, stop after 3 epochs without gain in validation PR-AUC, CPU, `torch.manual_seed(common.SEED)`, then refit on Jan to Sep for the chosen epoch count.
  - Pick the best model by validation PR-AUC. Never choose a threshold on validation.
  - Cut-off reported here is the placeholder "riskiest 20%"; Task 10 fixes the real one.
  - The final test section starts with the marker cell `# ---- FINAL TEST (run once) ----` and is the only place test rows are touched.
  - Explain with permutation importance (on valid) and errors by group, saying plainly whether any airline or airport is treated clearly worse (flag recall gaps above 10 points between groups of at least 2,000 flights).
- [ ] **Step 4: Convert and execute** (`06a`), expect exit 0.
- [ ] **Step 5: Run all tests,** expect PASS.
- [ ] **Step 6: Second brain** (results table, why the best model won, fairness note, calibration drift, `log.md`, tick the first-prediction lines of Task 6).
- [ ] **Step 7: Commit and push:** `git add -A && git commit -m "feat: notebook 06a first prediction models" && git push`

---

### Task 9: Notebook 06b, second prediction

**Files:**
- Create: `notebooks/06b_models_second.ipynb`
- Modify: `tests/test_outputs.py`, `second-brain/*`

**Interfaces:**
- Consumes: `features.parquet`, `reports/02_second_prediction.csv` (`choice`).
- Produces: `data/processed/predictions_test_second.parquet` (`row_id, pred, y_true`; test rows of flights that departed); `reports/06b_valid_results.csv`, `reports/06b_results.csv` (`model, metric, value`); `reports/06b_importance.csv`; `reports/06b_group_errors.csv`.
  - If `choice == "minutes_late"`: models `dumb_median, ridge, gradient_boosting`, metric `mae_minutes`, target `dep_delay`, departed flights only.
  - If `choice == "over_60_min"`: models `dumb_never, logistic_regression, gradient_boosting`, metric `pr_auc` (same fallback structure as 06a).

- [ ] **Step 1: Write the failing test**

```python
def test_task6b_second_prediction():
    choice = _csv("02_second_prediction.csv").loc[0, "choice"]
    r = _csv("06b_results.csv")
    if choice == "minutes_late":
        mae = r[r["metric"] == "mae_minutes"].set_index("model")["value"]
        assert {"dumb_median", "ridge", "gradient_boosting"} <= set(mae.index)
        assert mae["ridge"] < mae["dumb_median"] and mae["gradient_boosting"] < mae["dumb_median"]
    else:
        pr = r[r["metric"] == "pr_auc"].set_index("model")["value"]
        assert pr["logistic_regression"] > pr["dumb_never"] and pr["gradient_boosting"] > pr["dumb_never"]
    p = pd.read_parquet(common.PROC / "predictions_test_second.parquet")
    assert p["row_id"].is_unique and len(p) > 90_000
```

- [ ] **Step 2: Run to see it fail.** Expected FAIL (file not found).
- [ ] **Step 3: Build `06b_models_second`.** Same section order, marker cell and 42-seed rule as Task 8, using the same features and the same leakage list (read `06_leakage_removed.csv`, do not rewrite it). For `minutes_late`: `Ridge` with the Task 8 preprocessing (alpha in `[0.1, 1, 10]`) and `HistGradientBoostingRegressor(loss="absolute_error")`, because the metric is MAE and delays have heavy tails; report the median alongside the mean error and show the effect of the largest delays. Permutation importance and errors by carrier, origin, dest_group and hour band.
- [ ] **Step 4: Convert and execute** (`06b`), expect exit 0.
- [ ] **Step 5: Run all tests,** expect PASS (including the test-opened-once guard for `06b`).
- [ ] **Step 6: Second brain** (results, `log.md`, tick the remaining Task 6 lines).
- [ ] **Step 7: Commit and push:** `git add -A && git commit -m "feat: notebook 06b second prediction models" && git push`

---

### Task 10: Notebook 07, decision cards

**Files:**
- Create: `notebooks/07_decision.ipynb`
- Modify: `tests/test_outputs.py`, `second-brain/*`

**Interfaces:**
- Consumes: both prediction files, `features.parquet` (for `part` and week), `destination_groups.parquet`, `reports/03_groups.csv`, `reports/06a_results.csv`, `reports/06b_results.csv`.
- Produces: `reports/07_cards.csv` (`card, line_no, line, text`; `card` is `first` or `second`, `line_no` 1 to 11); `reports/07_value_first.csv` (`share_flagged, flagged, hits, precision, recall, random_recall, lift`); `reports/07_value_second.csv` (`threshold_minutes, flagged, true_long, mae_model, mae_dumb, gain_minutes`); `reports/07_weekly_scores.csv` (`prediction, week_start, n, metric, value`).

- [ ] **Step 1: Write the failing contract test**

```python
def test_task7_cards():
    c = _csv("07_cards.csv")
    for card in ["first", "second"]:
        x = c[c["card"] == card]
        assert x["line_no"].tolist() == list(range(1, 12)) and x["text"].str.len().gt(15).all()
    assert not c["text"].str.contains(r"\{|\}|nan", case=False, regex=True).any()
    v1 = _csv("07_value_first.csv").iloc[0]
    assert v1["recall"] > v1["random_recall"] and v1["lift"] > 1
    v2 = _csv("07_value_second.csv").iloc[0]
    assert v2["mae_model"] < v2["mae_dumb"] and v2["gain_minutes"] > 0
    w = _csv("07_weekly_scores.csv")
    assert w.groupby("prediction")["week_start"].nunique().min() >= 12
```

- [ ] **Step 2: Run to see it fail.** Expected FAIL (file not found).
- [ ] **Step 3: Build `07_decision`** per spec §7 Task 7. Pinned:
  - First card cut-off: the riskiest 20% of test flights (stated as the assumed warning capacity of the agency, one flight in five).
  - Second card cut-off: flag flights whose predicted delay is 45 minutes or more (assumed connection-check level). If `choice` is the fallback, adapt the card to the yes-or-no form.
  - Value measured on the test part: flagged, hits, share of all late-or-cancelled caught, against random picking of the same number; for the second, minutes closer than the dumb answer.
  - Weekly scores for the test weeks. Re-check trigger (decision D-014): below 80% of the median weekly recall for two weeks in a row.
  - Per-group actions use the group names from `03_groups.csv`.
  - Each card has the 11 lines of the sheet's example, filled by code from computed numbers (f-strings), with no invented prices.
- [ ] **Step 4: Convert and execute** (`07`), expect exit 0.
- [ ] **Step 5: Run all tests,** expect PASS.
- [ ] **Step 6: Second brain** (D-014, card numbers in `results.md`, `log.md`, tick Task 7).
- [ ] **Step 7: Commit and push:** `git add -A && git commit -m "feat: notebook 07 decision cards" && git push`

---

### Task 11: Checkpoint, slide outline for the owner (gate)

**Files:**
- Create: `second-brain/deck-outline.md`

**Interfaces:**
- Produces: `deck-outline.md`, a table with one row per slide: number, plain-language title that states the message, the number shown, source file and column in `reports/`, chart file.

- [ ] **Step 1: Draft the outline** (about 12 rows) covering, in this order: the business problem and 3 questions; the data and what it cannot tell us; the best findings (3 to 4 slides); the destination groups; the metrics to follow; the two predictions explained in plain words and how they were tested honestly; the two decision cards; limits and re-check plan; where AI helped and what was checked by hand. Every slide's number must exist in a `reports/` file now.
- [ ] **Step 2: Show the outline to the owner in chat and STOP.** Do not write any slide, HTML or chart for the deck.
- [ ] **Step 3: Wait for an explicit yes.** If the owner changes anything, edit `deck-outline.md` and show it again.
- [ ] **Step 4: Record and commit** after the yes: D-015 "deck outline approved by owner" with today's date in `decisions.md`, then `git add -A && git commit -m "docs: approved deck outline" && git push`

---

### Task 12: The stakeholder deck

**Files:**
- Create: `slides/build_deck.py`, `slides/deck.html`, `slides/deck.pdf`, `slides/source_map.csv`, `slides/assets/*.png`
- Modify: `tests/test_outputs.py`, `second-brain/*`

**Interfaces:**
- Consumes: the approved `second-brain/deck-outline.md` and the `reports/` files it names.
- Produces: `slides/deck.pdf`; `slides/source_map.csv` (`slide, text, file, column`, one row per number shown).

- [ ] **Step 1: Write the failing test**

```python
import re
BANNED = ["PR-AUC", "AUC", "k-means", "silhouette", "chi-square", "logistic", "gradient boosting",
          "MLP", "neural network", "hyperparameter", "permutation", "recall", "precision"]

def test_task8_deck():
    pdf = (common.ROOT / "slides" / "deck.pdf").read_bytes()
    assert 11 <= len(re.findall(rb"/Type\s*/Page(?!s)", pdf)) <= 14
    sm = pd.read_csv(common.ROOT / "slides" / "source_map.csv")
    assert len(sm) > 0 and all((common.REPORTS / f).exists() for f in sm["file"].unique())
    html = (common.ROOT / "slides" / "deck.html").read_text(encoding="utf-8")
    visible = re.sub(r"<style.*?</style>|<[^>]+>", " ", html, flags=re.S)
    assert not [w for w in BANNED if w.lower() in visible.lower()]
```

- [ ] **Step 2: Run to see it fail.** Expected FAIL (file not found).
- [ ] **Step 3: Build `slides/build_deck.py`.** It reads the numbers from the `reports/` files named in the outline (never hard-coded), draws charts with matplotlib into `slides/assets/`, writes `deck.html` (16:9, CSS `@page { size: 13.333in 7.5in; margin: 0 }`, one `section` per slide, big number plus one chart plus one sentence) and `source_map.csv`. Plain language: business words only, no jargon from the banned list; the AI-use slide states what AI did and what the owner checked, exactly as confirmed by the owner in Task 13 step 1 (if not yet confirmed, ask now).
- [ ] **Step 4: Export to PDF with Edge**

```bash
"C:\Program Files (x86)\Microsoft\Edge\Application\msedge.exe" --headless --disable-gpu --no-pdf-header-footer \
  --print-to-pdf="<repo>\slides\deck.pdf" "file:///<repo>/slides/deck.html"
```
Expected: `slides/deck.pdf` exists.
- [ ] **Step 5: Look at every page** with the Read tool (PDF pages). Fix overflow, cut-off text or unreadable charts; re-export.
- [ ] **Step 6: Run the tests,** expect PASS.
- [ ] **Step 7: Second brain** (tick Task 8 lines, `log.md`) and commit: `git add -A && git commit -m "feat: stakeholder deck" && git push`

---

### Task 13: README and final verification

**Files:**
- Modify: `README.md`, `second-brain/*`
- Create: none

- [ ] **Step 1: Ask the owner what they personally reviewed or checked**, in their own words. Write only that into the README and the AI-use slide.
- [ ] **Step 2: Write the full `README.md`:** what the project is; data source, licence and citation; the "Data" section (where to get the five CSVs, put them in `data/`); how to run (`pip install -r requirements.txt`, `python tools/run_notebooks.py`, `python -m pytest tests`); the order and purpose of notebooks; result summary with numbers copied from `reports/`; how AI tools were used and what the owner checked (from step 1).
- [ ] **Step 3: Fresh-clone check.** Clone the repo to a new empty temp directory, copy `data/*.csv` into its `data/`, run `python tools/run_notebooks.py` and `python -m pytest tests -v`.
Expected: every notebook exits 0; all tests PASS.
- [ ] **Step 4: Reproducibility check.** Run the notebook chain a second time in that clone and compare every `reports/*.csv` with the first run (values equal to 6 decimals; the neural network rows may differ by up to 0.005). Note any mismatch in `problems.md`.
- [ ] **Step 5: Hygiene and traceability.** `python -m pytest tests/test_repo_hygiene.py -v` passes; `git ls-files` shows no CSV outside `reports/` and `slides/`; every `done-when.md` box is ticked or has a one-line reason; every number on the slides appears in `source_map.csv`.
- [ ] **Step 6: Commit, push, and confirm**

```bash
git add -A && git commit -m "docs: final README and verification" && git push
gh repo view bssam99/nyc-flights-2023-delay-prediction --json url,visibility,pushedAt
```
Expected: `PUBLIC`, `pushedAt` is now. Report the repo URL, the test results and anything left undone to the owner.
