import os
import warnings
import sys

def configure_warnings():
    """Configure warning filters for the current process."""
    # Disable all DeprecationWarnings by default
    warnings.filterwarnings("ignore", category=DeprecationWarning)

    # Disable specific warnings
    warnings.filterwarnings("ignore", category=DeprecationWarning, module="pkg_resources")
    warnings.filterwarnings("ignore", category=DeprecationWarning, module="pygame.pkgdata")
    warnings.filterwarnings("ignore", message=".*declare_namespace.*")
    warnings.filterwarnings("ignore", message=".*pkg_resources is deprecated.*")
    warnings.filterwarnings("ignore", message=".*Deprecated call to `pkg_resources.declare_namespace.*")

    # Set environment variable to suppress warnings
    os.environ["PYTHONWARNINGS"] = "ignore::DeprecationWarning"

    # Force warnings to be ignored immediately
    if not sys.warnoptions:
        warnings.simplefilter("ignore", DeprecationWarning)
        warnings.simplefilter("ignore", ImportWarning)

# Configure warnings in the main process
configure_warnings()
