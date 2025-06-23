from pathlib import Path


def collect_py_files(root: Path, exclude_dirs: set[str]) -> list[Path]:
    files = []
    for path in root.rglob("*.py"):
        if any(part in exclude_dirs for part in path.parts):
            continue
        files.append(path)
    return files


def test_no_xcxc():
    root = Path(__file__).resolve().parents[1]
    exclude_dirs = {".venv", "build", "metta.egg-info", ".git"}
    py_files = collect_py_files(root, exclude_dirs)

    this_file = Path(__file__).resolve()

    offenders = []
    for file in py_files:
        if file == this_file:
            continue
        content = file.read_text(encoding="utf-8", errors="ignore")
        if "xcxc" in content:
            offenders.append(str(file))

    assert offenders == [], f"'xcxc' found in: {offenders}"
