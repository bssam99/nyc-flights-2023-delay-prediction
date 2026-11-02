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
