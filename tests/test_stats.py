import numpy as np

from src.stats import day_bootstrap, per_day_counts


def test_per_day_counts_aligns_days_and_fills_missing_days_with_zero():
    out = per_day_counts(["d1", "d1", "d2", "d3"], [1, 0, 1, 1], [True, True, True, False], ["d1", "d2", "d3"])
    assert out.tolist() == [[1.0, 2.0], [1.0, 1.0], [0.0, 0.0]]


def test_day_bootstrap_excludes_zero_for_a_real_gap_and_includes_it_for_no_gap():
    rng = np.random.default_rng(0)
    n = np.full(120, 200)
    a, b = np.column_stack([rng.binomial(n, 0.20), n]), np.column_stack([rng.binomial(n, 0.40), n])
    point, low, high = day_bootstrap(a, b, lambda x, y: (y - x) * 100, draws=500)
    assert low < point < high and low > 10            # the true gap is 20 points
    c, d = np.column_stack([rng.binomial(n, 0.30), n]), np.column_stack([rng.binomial(n, 0.30), n])
    _, low0, high0 = day_bootstrap(c, d, lambda x, y: (y - x) * 100, draws=500)
    assert low0 < 0 < high0


def test_day_bootstrap_is_wider_than_the_flight_level_interval_when_days_share_a_shock():
    rng = np.random.default_rng(1)
    n = np.full(100, 300)
    rate = np.clip(0.25 + rng.normal(0, 0.08, 100), 0.01, 0.99)      # a bad day hits every flight of that day
    counts = np.column_stack([rng.binomial(n, rate), n])
    _, low, high = day_bootstrap(counts, counts, lambda x, y: x, draws=500)
    p = counts[:, 0].sum() / counts[:, 1].sum()
    flight_level_width = 2 * 1.96 * np.sqrt(p * (1 - p) / counts[:, 1].sum())
    assert (high - low) > 2 * flight_level_width
