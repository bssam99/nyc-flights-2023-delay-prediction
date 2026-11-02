"""Feature builders that only use what is known 2 hours before the scheduled departure."""
import pandas as pd

from src import common

WEATHER_COLUMNS = ["temp", "wind_speed", "wind_gust", "visib", "precip", "pressure", "weather_filled"]


def add_congestion(f: pd.DataFrame) -> pd.DataFrame:
    """Add origin_hour_flights: how many flights are scheduled from the same airport in the same hour."""
    out = f.sort_values("row_id").reset_index(drop=True)
    out["origin_hour_flights"] = out.groupby(["origin", "time_hour"])["row_id"].transform("size").astype(int)
    return out


def add_lagged_weather(f: pd.DataFrame, w: pd.DataFrame, lead_hours: int = common.LEAD_HOURS) -> pd.DataFrame:
    """Attach the weather reading taken `lead_hours` before the flight's scheduled hour. Flights are never dropped."""
    lagged = w[["origin", "time_hour"] + WEATHER_COLUMNS].copy()
    lagged["time_hour"] = lagged["time_hour"] + pd.Timedelta(hours=lead_hours)
    out = f.merge(lagged, on=["origin", "time_hour"], how="left")
    if len(out) != len(f):
        raise ValueError("the weather table has repeated rows for one airport and hour")
    return out.sort_values("row_id").reset_index(drop=True)


def add_previous_flight(f: pd.DataFrame, lead_hours: int = common.LEAD_HOURS) -> pd.DataFrame:
    """Add prev_dep_delay and prev_known from the same plane's closest earlier flight on the same date.

    The earlier flight only counts if it had already left `lead_hours` before this flight's scheduled departure,
    because only then is its delay known at prediction time.
    """
    out = f.sort_values("row_id").reset_index(drop=True)
    s = out[["row_id", "tailnum", "sched_dep_dt", "dep_delay"]].sort_values(["tailnum", "sched_dep_dt"])
    group = s.groupby(["tailnum", s["sched_dep_dt"].dt.normalize()])
    prev_delay = group["dep_delay"].shift()
    prev_left_at = group["sched_dep_dt"].shift() + pd.to_timedelta(prev_delay, unit="m")
    known = prev_delay.notna() & (prev_left_at <= s["sched_dep_dt"] - pd.Timedelta(hours=lead_hours))
    s["prev_known"] = known.astype("int8")
    s["prev_dep_delay"] = prev_delay.where(known)
    return out.merge(s[["row_id", "prev_dep_delay", "prev_known"]], on="row_id", how="left")
