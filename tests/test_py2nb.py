from tools.py2nb import convert


def test_convert_splits_markdown_and_code():
    nb = convert("# %% [markdown]\n# Hello\n# %%\nx = 1\n")
    assert [c.cell_type for c in nb.cells] == ["markdown", "code"]
    assert nb.cells[0].source == "Hello" and nb.cells[1].source == "x = 1"
