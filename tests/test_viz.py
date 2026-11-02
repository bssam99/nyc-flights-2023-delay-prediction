import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from src import viz


def test_save_figure_writes_png_under_figures_and_returns_the_name(tmp_path, monkeypatch):
    monkeypatch.setattr(viz, "FIGURES", tmp_path)
    fig, ax = plt.subplots()
    ax.plot([1, 2, 3])
    assert viz.save_figure(fig, "demo.png") == "demo.png"
    assert (tmp_path / "demo.png").stat().st_size > 1000


def test_apply_style_makes_axes_quiet():
    viz.apply_style()
    fig, ax = plt.subplots()
    assert not ax.spines["top"].get_visible() and not ax.spines["right"].get_visible()
    plt.close(fig)
