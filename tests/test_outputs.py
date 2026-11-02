import pandas as pd

from src import common


def _csv(name):
    return pd.read_csv(common.REPORTS / name)


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
    g = pd.read_parquet(common.PROC / "destination_groups.parquet")
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
    f = pd.read_parquet(common.PROC / "features.parquet")
    assert len(f) == common.FLIGHT_ROWS and set(t["feature"]) <= set(f.columns)
    assert {"row_id", "part", "late_or_cancelled", "dep_delay"} <= set(f.columns)


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
    p = pd.read_parquet(common.PROC / "predictions_test_second.parquet")
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
    sm = pd.read_csv(common.ROOT / "slides" / "source_map.csv")
    assert list(sm.columns) == ["slide", "text", "file", "column"] and len(sm) > 0
    assert all((common.REPORTS / f).exists() for f in sm["file"].unique())
    visible = " ".join(sh.text_frame.text for slide in prs.slides for sh in slide.shapes if sh.has_text_frame)
    assert not [w for w in BANNED if w.lower() in visible.lower()]
