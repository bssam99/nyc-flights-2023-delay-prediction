import hashlib

import numpy as np
import pandas as pd
import pytest

from src import common


def _csv(name):
    return pd.read_csv(common.REPORTS / name)


def _parquet(name, **kwargs):
    common.require_processed(name)        # a clear "run the earlier notebooks first" message instead of a pandas error
    return pd.read_parquet(common.PROC / name, **kwargs)


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
    f = _parquet("flights_clean.parquet")
    assert len(f) == common.FLIGHT_ROWS and f["row_id"].is_unique and f["dep_time"].max() < 2400
    assert len(_parquet("weather_clean.parquet")) == 26_280


def test_task2_findings_and_second_prediction():
    f = _csv("02_findings.csv")
    assert len(f) == 5 and f["action"].str.len().gt(10).all()
    assert (f["check_type"] != "none").sum() >= 3
    assert set(f["check_type"]) <= {"confidence_interval", "chi_square", "rank_correlation", "none"}
    assert all((common.REPORTS / "figures" / p).exists() for p in f["chart_file"])
    strange = f[f["looks_strange_at_first"]]
    assert len(strange) >= 1 and strange["why"].str.len().gt(20).all()
    s = _csv("02_second_prediction.csv")
    assert len(s) == 1 and s.loc[0, "choice"] in {"minutes_late", "over_60_min"}


def test_task3_groups():
    g = _parquet("destination_groups.parquet")
    assert g["dest"].is_unique and len(g) == 118 and g["group_name"].notna().all()
    r = _csv("03_groups.csv")
    assert r[["group_name", "typical", "action"]].notna().all().all()
    assert r["size"].sum() == 118
    assert list(_csv("03_k_selection.csv")["k"]) == list(range(2, 11))


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


FORBIDDEN = {"dep_time", "dep_delay", "arr_time", "arr_delay", "air_time"}


def test_task5_features():
    t = _csv("05_feature_table.csv")
    assert len(t) >= 10 and (t["known_in_advance"] == "yes").all()
    assert t[["what", "why"]].notna().all().all() and not (set(t["feature"]) & FORBIDDEN)
    f = _parquet("features.parquet")
    assert len(f) == common.FLIGHT_ROWS and set(t["feature"]) <= set(f.columns)
    assert {"row_id", "part", "late_or_cancelled", "dep_delay"} <= set(f.columns)


def test_task6a_first_prediction():
    r = _csv("06a_results.csv")
    pr = r[r["metric"] == "pr_auc"].set_index("model")["value"]
    assert {"dumb_never_late", "hour_rule", "logistic_regression", "gradient_boosting",
            "neural_network"} <= set(pr.index)
    for m in ["logistic_regression", "gradient_boosting", "neural_network"]:
        assert pr[m] > pr["dumb_never_late"]
    p = _parquet("predictions_test_late.parquet")
    f = _parquet("features.parquet", columns=["row_id", "part"])
    assert set(p["row_id"]) == set(f.loc[f["part"] == "test", "row_id"]) and p["risk"].between(0, 1).all()
    assert {"dep_time", "dep_delay", "arr_time", "arr_delay", "air_time"} <= set(_csv("06_leakage_removed.csv")["column"])
    assert {"carrier", "origin", "dest_group", "hour_band"} <= set(_csv("06a_group_errors.csv")["dimension"])


def test_task6b_second_prediction():
    choice = _csv("02_second_prediction.csv").loc[0, "choice"]
    r = _csv("06b_results.csv")
    if choice == "minutes_late":
        mae = r[r["metric"] == "mae_minutes"].set_index("model")["value"]
        assert {"dumb_median", "ridge", "gradient_boosting"} <= set(mae.index)
        # every model is compared with the dumb answer; the model we use must beat it. Ridge does not on the held-out
        # months (see notebook 06b): that negative result is reported, not hidden.
        assert mae.drop("dumb_median").min() < mae["dumb_median"]
    else:
        pr = r[r["metric"] == "pr_auc"].set_index("model")["value"]
        assert pr["logistic_regression"] > pr["dumb_never"] and pr["gradient_boosting"] > pr["dumb_never"]
    p = _parquet("predictions_test_second.parquet")
    assert p["row_id"].is_unique and len(p) > 90_000


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


import re

BANNED = ["PR-AUC", "AUC", "k-means", "silhouette", "chi-square", "logistic", "gradient boosting",
          "MLP", "neural network", "hyperparameter", "permutation", "recall", "precision"]


def test_task8_deck():
    from pptx import Presentation
    pdf = (common.ROOT / "slides" / "deck.pdf").read_bytes()
    assert 11 <= len(re.findall(rb"/Type\s*/Page(?!s)", pdf)) <= 14
    prs = Presentation(common.ROOT / "slides" / "deck.pptx")
    assert 11 <= len(prs.slides) <= 14
    assert all(slide.has_notes_slide and slide.notes_slide.notes_text_frame.text.strip() for slide in prs.slides)
    assert prs.core_properties.last_modified_by != "Steve Canny" and prs.core_properties.title.strip()
    # no leftovers of python-pptx's built-in template (a 2013 date, "Microsoft Macintosh PowerPoint", a 4:3 page format)
    import datetime
    import zipfile
    assert prs.core_properties.created > datetime.datetime(2024, 1, 1) and prs.core_properties.modified > datetime.datetime(2024, 1, 1)
    app = zipfile.ZipFile(common.ROOT / "slides" / "deck.pptx").read("docProps/app.xml").decode("utf-8")
    assert "Macintosh" not in app and "4:3" not in app and "On-screen" not in app
    sm = pd.read_csv(common.ROOT / "slides" / "source_map.csv")
    assert list(sm.columns) == ["slide", "text", "file", "column"] and len(sm) > 0
    assert all((common.REPORTS / f).exists() for f in sm["file"].unique())
    visible = " ".join(sh.text_frame.text for slide in prs.slides for sh in slide.shapes if sh.has_text_frame)
    assert not [w for w in BANNED if w.lower() in visible.lower()]
    assert "never seen" not in visible.lower() and "safe default" not in visible.lower()
    assert "makes its next flight" not in visible.lower() and "delays its next flight" not in visible.lower()


def test_no_feature_reveals_a_cancelled_flight():
    f = _parquet("features.parquet")
    cancelled = f["dep_delay"].isna()
    for c in _csv("05_feature_table.csv")["feature"]:
        col = f[c]
        groups = [col.isna()] + ([col == v for v in col.dropna().unique()] if col.nunique() <= 5 else [])
        for mask in groups:
            if mask.sum() >= 500:
                assert cancelled[mask].mean() < 0.9, f"{c}: a group of {int(mask.sum())} flights is almost all cancelled"


def test_task2_second_prediction_records_the_probe_with_and_without_month():
    s = _csv("02_second_prediction.csv").iloc[0]
    assert {"probe_gain", "probe_gain_with_month"} <= set(s.index)
    assert (s["probe_gain"] >= 0.03) == (s["choice"] == "minutes_late")


def test_task2_key_numbers_match_the_finding_texts():
    """The deck reads these numbers by name instead of cutting them out of the finding sentences by position."""
    k = _csv("02_key_numbers.csv").astype(str).set_index("name")["value"]
    n = _csv("02_findings.csv")["number"]
    assert f"{k['morning_pct']}% of flights" in n[0] and f"against {k['evening_pct']}%" in n[0]
    assert f"left more than 15 minutes late, {k['late_plane_pct']}% of" in n[1] and f"against {k['ok_plane_pct']}% when" in n[1]
    assert f"{k['airline_best']} has {k['best_pct']}% of flights" in n[2] and f"{k['airline_worst']} has {k['worst_pct']}%" in n[2]
    assert f"adds {k['plane_points']} points and bad weather adds {k['weather_points']}" in n[3]
    assert f"{k['ten_days_pct']}% of all cancellations" in n[4]
    thin = _csv("03_groups.csv").query("group_id == 2").iloc[0]
    assert f"{int(thin['late_pct'])}% of flights late" in thin["typical"]


def test_task2_weather_finding_checks_the_comparison_its_title_makes():
    """The title says a late earlier flight raises the risk more than bad weather, so the check must be an interval for the
    difference between the two jumps (resampling whole days), not only for the weather jump."""
    f = _csv("02_findings.csv")
    row = f[f["finding"].str.contains("bigger jump in risk than bad weather")].iloc[0]
    assert "difference between the two jumps" in row["check_result"] and "95% interval" in row["check_result"]
    airlines = f[f["finding"].str.contains("Airlines differ")].iloc[0]
    assert "picked from the same data" in airlines["check_result"]


def test_task2_domino_gap_is_checked_within_the_same_day():
    d = _csv("02_domino_adjusted.csv").iloc[0]
    assert 0 < d["same_day_gap_points"] < d["raw_gap_points"] and d["strata"] > 1000
    f = _csv("02_findings.csv")
    assert "same day" in f.loc[1, "check_result"].lower()


def test_task7_card1_can_be_used_as_described():
    """The card must describe a rule that can run live: a cut-off taken from the scores of the previous 7 days (no future
    flights, no outcomes). Ranking each day needs the whole day ahead; ranking each hour on its own is the strict case."""
    v = _csv("07_value_first.csv").iloc[0]
    assert abs(v["share_live_rule"] - 0.20) < 0.03
    assert v["recall_live_rule"] > 1.5 * v["share_live_rule"]
    assert v["recall_ranked_per_hour"] > v["random_recall"]
    cards = _csv("07_cards.csv")
    text = " ".join(cards["text"]).lower()
    assert "tomorrow" not in text and "warn only if" not in text and "day by day" not in text
    cut_off = cards.query("card == 'first' and line == 'Cut-off'")["text"].iloc[0]
    assert f"{v['recall_live_rule']:.0%}" in cut_off and "previous 7 days" in cut_off and f"{v['recall_ranked_per_hour']:.0%}" in cut_off
    finding = cards.query("card == 'first' and line == 'Finding'")["text"].iloc[0]
    assert "flew earlier that day" not in finding and "left at least 2 hours" in finding


def test_task3_steady_routes_are_not_oversold():
    actions = _csv("03_groups.csv")["action"]
    assert not actions.str.contains("safe default", case=False).any()
    assert not actions.str.contains("no special warning", case=False).any()      # card 1 gives the same list to every destination


def test_readme_discloses_that_the_test_months_overall_rates_were_seen_in_design():
    readme = (common.ROOT / "README.md").read_text(encoding="utf-8")
    assert "D-004" in readme and "never seen" not in readme.lower()


def test_plane_columns_do_not_reveal_a_cancelled_flight_within_one_airline():
    """Nearly all no-tail flights are United's, and every one is cancelled. Planes drawn from other airlines' fleets would
    still stand out among United's own flights, so the plane columns alone must not separate cancelled from normal United
    flights (January-September). Drawing from all airlines gave AUC 0.90; a same-airline draw is near 0.5."""
    from sklearn.ensemble import HistGradientBoostingClassifier
    from sklearn.metrics import roc_auc_score
    f = _parquet("features.parquet")
    united = f.query("carrier == 'UA' and part != 'test'").sample(frac=1.0, random_state=common.SEED)
    cancelled = united["dep_delay"].isna().astype(int)
    half = len(united) // 2
    model = HistGradientBoostingClassifier(random_state=common.SEED).fit(united[["seats", "plane_age"]].iloc[:half], cancelled.iloc[:half])
    auc = roc_auc_score(cancelled.iloc[half:], model.predict_proba(united[["seats", "plane_age"]].iloc[half:])[:, 1])
    assert auc < 0.7, f"the plane columns separate cancelled United flights from the rest (AUC {auc:.2f})"


def test_parquet_helper_gives_a_clear_message_when_a_notebook_has_not_run(tmp_path, monkeypatch):
    monkeypatch.setattr(common, "PROC", tmp_path)
    with pytest.raises(FileNotFoundError, match="Run the earlier notebooks"):
        _parquet("features.parquet")


def test_original_files_are_unchanged_since_notebook_01():
    common.require_raw_files()
    fingerprint = _csv("01_raw_fingerprint.csv")
    assert len(fingerprint) == 5
    for r in fingerprint.itertuples():
        assert hashlib.sha256((common.RAW / r.file).read_bytes()).hexdigest() == r.sha256, f"{r.file} changed since notebook 01 ran"


def test_task3_groups_are_checked_for_stability_and_have_a_tuning_version():
    st = _csv("03_stability.csv")
    assert list(st.columns) == ["check", "value"] and len(st) >= 6
    g = _parquet("destination_groups_fit.parquet")
    assert g["dest"].is_unique and len(g) == 118


def test_tuning_uses_groups_built_without_the_valid_months():
    assert "dest_group_fit" in _parquet("features.parquet").columns
    # the January-June groups call a destination "rare" by its January-June flights, the final groups by its January-September flights
    flights = _parquet("flights_clean.parquet", columns=["dest", "part"])
    for file, parts in [("destination_groups_fit.parquet", ["fit"]), ("destination_groups.parquet", ["fit", "valid"])]:
        groups = _parquet(file).set_index("dest")
        seen = flights[flights["part"].isin(parts)].groupby("dest").size().reindex(groups.index).fillna(0)
        assert ((groups["group_id"] == -1) == (seen < 100)).all(), f"{file}: the rare destinations do not follow the flights of {parts}"
    import nbformat
    for name in ["06a_models_first", "06b_models_second"]:
        cells = [c.source for c in nbformat.read(common.ROOT / "notebooks" / f"{name}.ipynb", 4).cells if c.cell_type == "code"]
        marker = next(i for i, c in enumerate(cells) if "# ---- FINAL TEST (run once) ----" in c)
        assert any("dest_group_fit" in c for c in cells[:marker]), f"{name}: tuning does not use the January-June groups"


def test_task7_recheck_line_comes_from_the_validation_months():
    w = _csv("06a_valid_weekly_reach.csv")
    assert len(w) >= 10 and {"week_start", "n", "recall_at_20pct"} <= set(w.columns)
    line = _csv("07_value_first.csv").iloc[0]["recheck_line"]
    assert abs(line - 0.8 * w.loc[w["n"] >= 5000, "recall_at_20pct"].median()) < 0.001
    text = _csv("07_cards.csv").query("card == 'first' and line_no == 9")["text"].iloc[0]
    assert "July to September" in text and f"{line:.0%}" in text


def test_task2_intervals_resample_whole_days():
    f = _csv("02_findings.csv").iloc[:4]
    assert (f["check_type"] == "confidence_interval").all()
    assert f["check_result"].str.contains("whole days").all()


def test_task2_first_interval_can_be_recomputed_from_the_flights():
    """Evening (19-21 h) against morning (6-8 h) in January-September, resampling whole days: the numbers in the finding must be these."""
    from src.stats import day_bootstrap, per_day_counts
    flights = _parquet("flights_clean.parquet", columns=["part", "hour", "sched_dep_dt", "late_or_cancelled"])
    train = flights[flights["part"] != "test"]
    days = np.array(sorted(train["sched_dep_dt"].dt.date.unique()), dtype=object)
    dates, late = train["sched_dep_dt"].dt.date.to_numpy(), train["late_or_cancelled"].to_numpy()
    morning = per_day_counts(dates, late, train["hour"].between(6, 8).to_numpy(), days)
    evening = per_day_counts(dates, late, train["hour"].between(19, 21).to_numpy(), days)
    ratio, low, high = day_bootstrap(morning, evening, lambda a, b: b / a)
    assert f"evening is {ratio:.1f} times the morning rate (95% interval {low:.1f} to {high:.1f})" in _csv("02_findings.csv").loc[0, "check_result"]


def test_cards_have_no_time_s_wording():
    assert not _csv("07_cards.csv")["text"].str.contains(r"time\(s\)", regex=True).any()


def test_owner_review_statement_says_which_version_was_checked():
    """The owner's checks were reported for earlier versions. Until the owner re-checks the final one (the flag in the
    checklist), the README, the checklist and the slide must say so, and the test count they quote must be the real one."""
    checklist = (common.ROOT / "second-brain" / "review-checklist.md").read_text(encoding="utf-8")
    readme = (common.ROOT / "README.md").read_text(encoding="utf-8")
    flag = re.search(r"\*\*Re-check on the final version:\*\* (pending|done)", checklist)
    assert flag, "the checklist must carry a line '**Re-check on the final version:** pending' or 'done'"
    count = sum(len(re.findall(r"^def test_", p.read_text(encoding="utf-8"), flags=re.M)) for p in (common.ROOT / "tests").glob("test_*.py"))
    assert f"then `{count} passed`" in checklist and f"# {count} tests" in readme, f"quote {count} tests"
    from pptx import Presentation
    slide12 = " ".join(sh.text_frame.text for sh in Presentation(common.ROOT / "slides" / "deck.pptx").slides[11].shapes if sh.has_text_frame)
    if flag.group(1) == "pending":
        assert "earlier version" in readme and "not yet repeated" in readme
        assert "earlier version of the project" in slide12
    else:
        assert "earlier version of the project" not in slide12


def test_readme_names_the_python_version_the_pinned_packages_need():
    readme = (common.ROOT / "README.md").read_text(encoding="utf-8")
    assert re.search(r"Python 3\.11 or newer", readme), "pandas 3 and numpy 2.4 need Python 3.11 or later; the README must say so"


def test_readme_numbers_match_the_report_files():
    readme = (common.ROOT / "README.md").read_text(encoding="utf-8")
    v1, v2 = _csv("07_value_first.csv").iloc[0], _csv("07_value_second.csv").iloc[0]
    cal = _csv("06a_calibration.csv")
    predicted = (cal["mean_predicted"] * cal["flights"]).sum() / cal["flights"].sum()
    hours = _csv("06a_group_errors.csv").query("dimension == 'hour_band' and metric == 'recall'").sort_values("value")
    adj = _csv("02_domino_adjusted.csv").iloc[0]
    expected = [f"reaches **{v1.recall:.0%}**", f"{v1.precision * 100:.0f} warnings in 100", f"{v1.base_rate * 100:.0f} flights in 100 are late",
                f"{v2.mae_model:.2f} minutes against {v2.mae_dumb:.2f}", f"({v2.lift:.1f} times more)",
                f"({predicted:.0%} predicted, {v1.base_rate:.0%} real)", f"only {hours.iloc[0].value:.0%} of the time (evening: {hours.iloc[-1].value:.0%})",
                f"{adj.same_day_gap_points:.0f} points of that gap remain"]
    expected += [f"reaches **{v1.recall_live_rule:.0%}** of the late or cancelled flights (from", f"would reach {v1.recall_ranked_per_hour:.0%}"]
    st = _csv("03_stability.csv")["value"]
    expected += [f"removing the time-of-day columns ({st.iloc[2]:.2f}) or using only January-June ({st.iloc[4]:.2f})"]
    missing = [e for e in expected if e not in readme]
    assert not missing, f"README numbers are out of date: {missing}"
