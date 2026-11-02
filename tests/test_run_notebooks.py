import nbformat
import pytest

from tools.run_notebooks import run_notebook


def _nb(path, code):
    nb = nbformat.v4.new_notebook(cells=[nbformat.v4.new_code_cell(code)])
    nbformat.write(nb, path)


def test_run_notebook_stores_output(tmp_path):
    p = tmp_path / "a.ipynb"
    _nb(p, "print('hi')")
    run_notebook(p)
    assert "hi" in nbformat.read(p, 4).cells[0].outputs[0].text


def test_run_notebook_raises_on_error(tmp_path):
    p = tmp_path / "b.ipynb"
    _nb(p, "1/0")
    with pytest.raises(Exception):
        run_notebook(p)
