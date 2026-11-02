"""Run notebooks in place from a fresh kernel, in filename order. Usage: python tools/run_notebooks.py [01 02 ...]"""
import sys
import time
from pathlib import Path

import nbformat
from nbconvert.preprocessors import ExecutePreprocessor

NOTEBOOKS = Path(__file__).resolve().parents[1] / "notebooks"

if sys.platform == "win32":  # zmq (the kernel link) needs the selector loop on Windows
    import asyncio
    asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())


def run_notebook(path: Path, timeout: int = 3600) -> None:
    """Execute every cell with a fresh kernel and store the outputs. Raises on the first cell error."""
    path = Path(path)
    nb = nbformat.read(path, 4)
    ExecutePreprocessor(timeout=timeout, kernel_name="python3").preprocess(nb, {"metadata": {"path": str(path.parent)}})
    nbformat.write(nb, path)


def main(argv: list) -> int:
    prefixes = argv or [""]
    for prefix in prefixes:
        if not any(p.name.startswith(prefix) for p in NOTEBOOKS.glob("*.ipynb")):
            print(f"no notebook starts with '{prefix}'")
            return 1
    for path in sorted(NOTEBOOKS.glob("*.ipynb")):
        if any(path.name.startswith(prefix) for prefix in prefixes):
            start = time.time()
            try:
                run_notebook(path)
            except Exception as error:  # report which notebook failed, then stop
                print(f"FAILED {path.name}: {error}")
                return 1
            print(f"ok {path.name} ({time.time() - start:.0f}s)")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
