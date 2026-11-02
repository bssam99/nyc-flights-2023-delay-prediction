"""Turn a '# %%' percent-format script into a notebook. The .ipynb is the deliverable."""
import sys
from pathlib import Path

import nbformat


def convert(source: str) -> nbformat.NotebookNode:
    """'# %% [markdown]' starts a markdown cell (lines lose their leading '# '); '# %%' starts a code cell."""
    cells, kind, buf = [], None, []

    def flush():
        text = "\n".join(buf).strip("\n")
        if kind and text:
            make = nbformat.v4.new_markdown_cell if kind == "markdown" else nbformat.v4.new_code_cell
            cells.append(make(text))

    for line in source.splitlines():
        if line.startswith("# %%"):
            flush()
            kind, buf = ("markdown" if "[markdown]" in line else "code"), []
        elif kind == "markdown":
            buf.append(line[2:] if line.startswith("# ") else ("" if line == "#" else line))
        elif kind:
            buf.append(line)
    flush()

    nb = nbformat.v4.new_notebook(cells=cells)
    nb.metadata["kernelspec"] = {"name": "python3", "display_name": "Python 3", "language": "python"}
    return nb


if __name__ == "__main__":
    src, dst = Path(sys.argv[1]), Path(sys.argv[2])
    nbformat.write(convert(src.read_text(encoding="utf-8")), dst)
    print(f"wrote {dst}")
