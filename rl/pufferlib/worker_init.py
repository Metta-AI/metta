# This file is imported before anything else in worker processes
import os
import warnings
import sys

# Configure warning filters
warnings.filterwarnings('ignore', category=DeprecationWarning, module='pkg_resources')
warnings.filterwarnings('ignore', message='.*declare_namespace.*', category=DeprecationWarning)

# Set environment variable
os.environ["PYTHONWARNINGS"] = "ignore::DeprecationWarning:pkg_resources"
