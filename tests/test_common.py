import pandas as pd
import pytest

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
