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
