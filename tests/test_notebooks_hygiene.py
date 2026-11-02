import re
from pathlib import Path

import nbformat

ROOT = Path(__file__).resolve().parents[1]
NOTEBOOKS = sorted((ROOT / "notebooks").glob("*.ipynb"))
LOCAL_PATH = re.compile(r"[A-Za-z]:(\\\\|\\|/)Users")


def _code(nb):
    return [c for c in nb.cells if c.cell_type == "code"]


def test_notebooks_ran_without_errors():
    for p in NOTEBOOKS:
        for c in _code(nbformat.read(p, 4)):
            assert c.execution_count is not None, f"{p.name}: a code cell was never executed"
            assert not [o for o in c.outputs if o.output_type == "error"], f"{p.name}: error output"


def test_notebook_outputs_are_small():
    for p in NOTEBOOKS:
        for c in _code(nbformat.read(p, 4)):
            for o in c.outputs:
                text = o.get("text") or o.get("data", {}).get("text/plain", "")
                text = "".join(text) if isinstance(text, list) else text
                assert len(text) <= 6000, f"{p.name}: an output is over 6,000 characters"


def test_notebooks_explain_themselves():
    for p in NOTEBOOKS:
        nb = nbformat.read(p, 4)
        code = _code(nb)
        markdown = [c for c in nb.cells if c.cell_type == "markdown"]
        assert len(markdown) * 2 >= len(code), f"{p.name}: too few markdown cells next to the code"
        assert "from src import common" in code[0].source, f"{p.name}: first code cell is not the setup cell"


def test_notebooks_have_no_local_paths():
    for p in NOTEBOOKS:
        assert not LOCAL_PATH.search(p.read_text(encoding="utf-8")), f"{p.name}: contains a local user path"


MARKER = "# ---- FINAL TEST (run once) ----"


def _cells_using_test_part_before_marker(nb):
    """Indices of code cells that mention the 'test' part before the cell holding the marker (None if no marker)."""
    cells = [c.source for c in nb.cells if c.cell_type == "code"]
    first = next((i for i, s in enumerate(cells) if MARKER in s), None)
    if first is None:
        return None
    return [i for i in range(first) if "'test'" in cells[i] or '"test"' in cells[i]]


def _nb(*sources):
    return nbformat.v4.new_notebook(cells=[nbformat.v4.new_code_cell(s) for s in sources])


def test_guard_flags_a_notebook_that_touches_the_test_part_early():
    assert _cells_using_test_part_before_marker(_nb("x = df[df.part == 'test']", MARKER)) == [0]
    assert _cells_using_test_part_before_marker(_nb("x = 1", MARKER, "y = df[df.part == 'test']")) == []
    assert _cells_using_test_part_before_marker(_nb("x = 1")) is None


def test_model_notebooks_open_the_test_part_only_after_the_marker():
    for p in NOTEBOOKS:
        if p.name.startswith(("06a", "06b")):
            early = _cells_using_test_part_before_marker(nbformat.read(p, 4))
            assert early is not None, f"{p.name}: no FINAL TEST marker cell"
            assert early == [], f"{p.name}: the test part is used before the marker (code cells {early})"
