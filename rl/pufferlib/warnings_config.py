import os
import warnings

# Configure warning filters
warnings.filterwarnings('ignore', category=DeprecationWarning, module='pkg_resources')
warnings.filterwarnings('ignore', message='.*declare_namespace.*', category=DeprecationWarning)

# Also set environment variable as backup
os.environ["PYTHONWARNINGS"] = "ignore::DeprecationWarning:pkg_resources"
