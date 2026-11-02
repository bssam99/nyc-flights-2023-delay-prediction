import numpy as np
import pandas as pd

from src.features import add_congestion, add_lagged_weather, add_previous_flight


def test_previous_flight_known_only_if_left_before_prediction_time():
    rows = [(0, "N1", "2023-05-01 06:00", 30.0), (1, "N1", "2023-05-01 08:00", 60.0),
            (2, "N1", "2023-05-01 09:30", 0.0), (3, "N1", "2023-05-01 12:00", 10.0),
            (4, "N2", "2023-05-01 12:00", 99.0), (5, "N1", "2023-05-02 06:00", -5.0),
            (6, "N3", "2023-05-01 08:00", np.nan), (7, "N3", "2023-05-01 12:00", 1.0),
            (8, None, "2023-05-01 08:00", 5.0), (9, None, "2023-05-01 12:00", 5.0)]
    f = pd.DataFrame(rows, columns=["row_id", "tailnum", "sched_dep_dt", "dep_delay"])
    f["sched_dep_dt"] = pd.to_datetime(f["sched_dep_dt"])
    out = add_previous_flight(f).set_index("row_id")
    assert len(out) == 10
    assert out["prev_known"].tolist() == [0, 0, 0, 1, 0, 0, 0, 0, 0, 0]
    assert out.loc[3, "prev_dep_delay"] == 0.0      # prev left 09:30, decision time 10:00
    assert out.loc[[0, 1, 2, 4, 5, 6, 7, 8, 9], "prev_dep_delay"].isna().all()


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
