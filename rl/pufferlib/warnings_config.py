import os
import warnings
import sys

def configure_warnings():
    """Configure warning filters for the current process."""
    # Configure warning filters for pkg_resources deprecation warnings
    warnings.filterwarnings('ignore', category=DeprecationWarning, module='pkg_resources')
    warnings.filterwarnings('ignore', message='.*declare_namespace.*', category=DeprecationWarning)
    warnings.filterwarnings('ignore', message='.*pkg_resources is deprecated.*', category=DeprecationWarning)
    warnings.filterwarnings('ignore', message='.*Deprecated call to `pkg_resources.declare_namespace.*', category=DeprecationWarning)

    # Configure warning filters for specific modules
    warnings.filterwarnings('ignore', category=DeprecationWarning, module='pygame.pkgdata')
    warnings.filterwarnings('ignore', category=DeprecationWarning, module='google')

    # Set environment variables for warning suppression
    os.environ["PYTHONWARNINGS"] = "ignore::DeprecationWarning:pkg_resources,ignore::DeprecationWarning:pygame.pkgdata,ignore::DeprecationWarning:google"

    # Force warnings to be ignored immediately
    if not sys.warnoptions:
        warnings.simplefilter("ignore", DeprecationWarning)

# Configure warnings in the main process
configure_warnings()
