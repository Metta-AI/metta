# metta/__init__.py
import sys
from pathlib import Path
from pkgutil import extend_path

metta_parent = str(Path(__file__).parent.parent.resolve())  # .../GitHub/metta
mettagrid_parent = str(Path(__file__).parent.parent / "mettagrid" / "src")  # .../GitHub/metta/mettagrid/src

for p in (metta_parent, mettagrid_parent):
    if p in sys.path:
        sys.path.remove(p)
sys.path.insert(0, metta_parent)  # repo root first
sys.path.append(mettagrid_parent)  # de-prioritize mettagrid/src

__path__ = extend_path(__path__, __name__)
