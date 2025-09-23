"""Test configuration ensuring local modules are imported from this workspace."""

from __future__ import annotations

import sys
from pathlib import Path

# Ensure the repository root is first on sys.path so local modules win over any
# globally installed copies (e.g., a sibling workspace checkout).
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

# Drop any pre-imported metta/mettagrid modules so future imports pick up the
# versions from the inserted repo root.
_PREFIXES = ("metta", "mettagrid")
for name in list(sys.modules):
    if name.startswith(_PREFIXES):
        sys.modules.pop(name)
