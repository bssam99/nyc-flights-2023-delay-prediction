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
