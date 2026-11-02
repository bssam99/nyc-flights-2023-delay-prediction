"""Rules shared by every notebook: paths, seed, the label and the time split."""
from pathlib import Path

import numpy as np
import pandas as pd

SEED = 42
LEAD_HOURS = 2        # we predict 2 hours before the scheduled departure
LATE_MINUTES = 15     # late = left more than 15 minutes after schedule
FLIGHT_ROWS = 435_352

ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / "data"
PROC = RAW / "processed"
REPORTS = ROOT / "reports"

RAW_FILES = ["flights.csv", "weather.csv", "planes.csv", "airports.csv", "airlines.csv"]

# which notebook writes each processed file, for clear "run this first" errors
WRITTEN_BY = {
    "flights_clean.parquet": "01_clean",
    "weather_clean.parquet": "01_clean",
    "planes_clean.parquet": "01_clean",
    "destination_groups.parquet": "03_groups",
    "features.parquet": "05_features",
    "predictions_test_late.parquet": "06a_models_first",
    "predictions_test_second.parquet": "06b_models_second",
}


def require_raw_files() -> None:
    """Stop with a clear message if a raw CSV is missing from data/."""
    missing = [name for name in RAW_FILES if not (RAW / name).exists()]
    if missing:
        raise FileNotFoundError(
            f"Missing raw data file(s): {', '.join(missing)} in {RAW}. "
            "Put the five CSV files in the data/ folder (see README, section 'Data')."
        )


def require_processed(*names: str) -> None:
    """Stop with a clear message if an earlier notebook has not written a file yet."""
    missing = [n for n in names if not (PROC / n).exists()]
    if missing:
        what = "; ".join(f"{n} (written by notebook {WRITTEN_BY.get(n, 'an earlier notebook')})" for n in missing)
        raise FileNotFoundError(
            f"Missing processed file(s): {what}. Run the earlier notebooks first: python tools/run_notebooks.py"
        )


def make_late_label(df: pd.DataFrame) -> pd.Series:
    """1 if the flight never left (no dep_time) or left more than 15 minutes late."""
    return (df["dep_time"].isna() | (df["dep_delay"] > LATE_MINUTES)).astype("int8")


def split_part(month: pd.Series) -> pd.Series:
    """fit = Jan-Jun, valid = Jul-Sep, test = Oct-Dec. Train is fit + valid."""
    values = np.select([month <= 6, month <= 9], ["fit", "valid"], default="test")
    return pd.Series(values, index=month.index)
