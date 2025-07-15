import importlib
from pathlib import Path


def import_all_modules_from_subpackage(package_name: str, subpackage: str) -> None:
    """Import all Python modules from a subpackage directory.

    This is useful for auto-registering modules that use decorators.
    Works with PEP 420 namespace packages.

    Args:
        package_name: The parent package name (e.g., 'metta.setup')
        subpackage: The subpackage name (e.g., 'components')
    """
    # Since we're in metta/setup/utils.py, we can use relative path
    current_file = Path(__file__)
    setup_dir = current_file.parent
    subpackage_path = setup_dir / subpackage

    if not subpackage_path.exists():
        return

    # Import all Python files in the subpackage
    for module_file in subpackage_path.glob("*.py"):
        if module_file.stem != "__init__" and not module_file.stem.startswith("_"):
            module_name = f"{package_name}.{subpackage}.{module_file.stem}"
            try:
                _ = importlib.import_module(module_name)
            except ImportError:
                pass
