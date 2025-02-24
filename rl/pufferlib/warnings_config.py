import os
import warnings
import sys

def configure_warnings():
    """Configure warning filters for the current process."""
    if not sys.warnoptions:  # If no warning filters are already set
        # Configure warning filters
        warnings.filterwarnings('ignore', category=DeprecationWarning, module='pkg_resources')
        warnings.filterwarnings('ignore', message='.*declare_namespace.*', category=DeprecationWarning)

        # Set environment variable
        os.environ["PYTHONWARNINGS"] = "ignore::DeprecationWarning:pkg_resources"

# Configure warnings in the main process
configure_warnings()
