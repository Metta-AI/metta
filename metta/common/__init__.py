"""Namespace bridge for the shared `metta.common` package.

The real implementation lives in the `metta-common` workspace project, but this
repository also contains a minimal `metta.common` package (for test helpers).
To allow both locations to coexist we need to explicitly extend the import
search path. Otherwise Python would stop at this directory and fail to find the
modules provided by `metta-common` (e.g., `metta.common.tool.run_tool`).
"""

from pathlib import Path
from pkgutil import extend_path

__path__ = extend_path(__path__, __name__)

# When running from a source checkout we also want to include the editable
# `metta-common` project under `common/src`. The editable finder installed by uv
# does not expose that path to pkgutil by default, so we append it manually.
_repo_root = Path(__file__).resolve().parents[2]
_common_src = _repo_root / "common" / "src" / "metta" / "common"
if _common_src.exists():
    common_path = str(_common_src)
    if common_path not in __path__:
        __path__.append(common_path)
