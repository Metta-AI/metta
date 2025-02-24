# Configure warnings before any other imports
import os
import sys
import warnings

# Disable all DeprecationWarnings by default
warnings.filterwarnings("ignore", category=DeprecationWarning)

# Specifically ignore pkg_resources related warnings
warnings.filterwarnings("ignore", category=DeprecationWarning, module="pkg_resources")
warnings.filterwarnings("ignore", category=DeprecationWarning, module="pygame.pkgdata")
warnings.filterwarnings("ignore", message=".*declare_namespace.*")
warnings.filterwarnings("ignore", message=".*pkg_resources is deprecated.*")
warnings.filterwarnings("ignore", message=".*Deprecated call to `pkg_resources.declare_namespace.*")

# Set environment variable to suppress warnings
os.environ["PYTHONWARNINGS"] = "ignore::DeprecationWarning"

# Only proceed with other imports after warnings are configured
from . import warnings_config
from . import worker_init
