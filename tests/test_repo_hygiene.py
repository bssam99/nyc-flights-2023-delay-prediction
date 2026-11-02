import re, subprocess
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SECRET = re.compile(r"(sk-[A-Za-z0-9]{20,}|gh[pousr]_[A-Za-z0-9]{30,}|AIza[0-9A-Za-z_\-]{30,})")

def _tracked():
    out = subprocess.run(["git", "ls-files"], cwd=ROOT, capture_output=True, text=True, check=True).stdout
    return out.split("\n")[:-1]

def test_no_data_keys_or_spreadsheets_tracked():
    bad = [f for f in _tracked() if f.startswith("data/") or f == ".env" or f.endswith(".xlsx")
           or (f.endswith(".csv") and not f.startswith("reports/") and not f.startswith("slides/"))]
    assert bad == []

def test_gitignore_has_required_entries():
    lines = (ROOT / ".gitignore").read_text().split("\n")
    for entry in ["data/", ".env", "ML_week_tasks.xlsx"]:
        assert entry in lines

def test_no_secret_patterns_in_tracked_text_files():
    hits = [f for f in _tracked() if f.endswith((".py", ".md", ".txt", ".ipynb", ".html", ".csv"))
            and SECRET.search((ROOT / f).read_text(errors="ignore"))]
    assert hits == []
