"""Uncertainty that respects the day: flights on one day share weather and knock-on delays, so we resample whole days."""
import numpy as np
import pandas as pd


def per_day_counts(dates, events, mask, all_days) -> np.ndarray:
    """(events, flights) per day for the rows selected by `mask`, aligned to `all_days` (a day without such flights counts 0)."""
    keep = np.asarray(mask, dtype=bool)
    sub = pd.DataFrame({"day": np.asarray(dates, dtype=object)[keep], "event": np.asarray(events, dtype=float)[keep]})
    counts = sub.groupby("day")["event"].agg(["sum", "size"]).reindex(list(all_days), fill_value=0)
    return counts.to_numpy(dtype=float)


def day_bootstrap_cells(cells, statistic, draws: int = 1000, seed: int = 42):
    """95% interval for statistic(rates) by resampling whole days with replacement, for any number of groups.

    cells: list of arrays of shape (days, 2) with (events, flights) per day, one per group, from `per_day_counts`.
    statistic: function of the list of group rates (one rate per group), returning a number (or an array, one per resample).
    Returns (point estimate, low, high). Every group is resampled on the same days.
    """
    arrays = [np.asarray(c, dtype=float) for c in cells]
    if any(a.shape != arrays[0].shape for a in arrays) or arrays[0].ndim != 2 or arrays[0].shape[1] != 2:
        raise ValueError("every group needs counts of the same shape (days, 2): (events, flights) per day")
    picks = np.random.default_rng(seed).integers(0, len(arrays[0]), size=(draws, len(arrays[0])))
    resampled = [a[picks, 0].sum(axis=1) / a[picks, 1].sum(axis=1) for a in arrays]
    point = statistic([a[:, 0].sum() / a[:, 1].sum() for a in arrays])
    low, high = np.percentile(statistic(resampled), [2.5, 97.5])
    return float(point), float(low), float(high)


def day_bootstrap(counts_a, counts_b, statistic, draws: int = 1000, seed: int = 42):
    """95% interval for statistic(rate_a, rate_b) by resampling whole days with replacement.

    counts_a, counts_b: arrays of shape (days, 2) with (events, flights) per day for two groups, from `per_day_counts`.
    Returns (point estimate, low, high).
    """
    return day_bootstrap_cells([counts_a, counts_b], lambda rates: statistic(rates[0], rates[1]), draws, seed)
