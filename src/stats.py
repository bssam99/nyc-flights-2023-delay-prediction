"""Uncertainty that respects the day: flights on one day share weather and knock-on delays, so we resample whole days."""
import numpy as np
import pandas as pd


def per_day_counts(dates, events, mask, all_days) -> np.ndarray:
    """(events, flights) per day for the rows selected by `mask`, aligned to `all_days` (a day without such flights counts 0)."""
    keep = np.asarray(mask, dtype=bool)
    sub = pd.DataFrame({"day": np.asarray(dates, dtype=object)[keep], "event": np.asarray(events, dtype=float)[keep]})
    counts = sub.groupby("day")["event"].agg(["sum", "size"]).reindex(list(all_days), fill_value=0)
    return counts.to_numpy(dtype=float)


def day_bootstrap(counts_a, counts_b, statistic, draws: int = 1000, seed: int = 42):
    """95% interval for statistic(rate_a, rate_b) by resampling whole days with replacement.

    counts_a, counts_b: arrays of shape (days, 2) with (events, flights) per day for two groups, from `per_day_counts`.
    Returns (point estimate, low, high).
    """
    a, b = np.asarray(counts_a, dtype=float), np.asarray(counts_b, dtype=float)
    assert a.shape == b.shape and a.ndim == 2 and a.shape[1] == 2
    picks = np.random.default_rng(seed).integers(0, len(a), size=(draws, len(a)))
    rate_a = a[picks, 0].sum(axis=1) / a[picks, 1].sum(axis=1)
    rate_b = b[picks, 0].sum(axis=1) / b[picks, 1].sum(axis=1)
    point = statistic(a[:, 0].sum() / a[:, 1].sum(), b[:, 0].sum() / b[:, 1].sum())
    low, high = np.percentile(statistic(rate_a, rate_b), [2.5, 97.5])
    return float(point), float(low), float(high)
