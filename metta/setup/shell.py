import logging
import sys
from functools import partial

import IPython
from traitlets.config import Config as IPythonConfig

from metta.common.util.console_messages import header, info, success, warning
from metta.common.util.fs import get_repo_root

sys.path.insert(0, str(get_repo_root() / "tools"))
from validate_config import load_and_print_config  # type: ignore

__name__ = "__ipython__"
logger = logging.getLogger(__name__)

REPO_ROOT = get_repo_root()
CONFIGS_DIR = REPO_ROOT / "configs"

load_cfg = partial(load_and_print_config, exit_on_failure=False, print_cfg=False)


def help_configs() -> None:
    header("=== Usage Examples ===")
    success("# Load configs:")
    info('cfg = load_cfg("sim_job.yaml")')
    info('cfg = load_cfg("trainer/trainer.yaml")')
    success("# Load configs with overrides:")
    info('cfg = load_cfg("train_job.yaml", ["trainer.curriculum=/env/mettagrid/arena/advanced"])')


# Create a new IPython config object
ipython_config = IPythonConfig()
ipython_config.InteractiveShellApp.extensions = ["autoreload"]
ipython_config.InteractiveShellApp.exec_lines = [
    "%autoreload 2",
    "success('Autoreload enabled: modules will be reloaded automatically when changed.')",
    "info('Use help_configs() to see available configurations')",
    "info('Use load_cfg() to load a configuration')",
]


try:
    personal_dir = str(REPO_ROOT / "personal")
    if personal_dir not in sys.path:
        sys.path.insert(0, personal_dir)
    from personal.shell_startup import *  # noqa

    success("Personal shell startup loaded successfully")
except ImportError:
    info("You can add custom shell startup code (e.g. importing functions you often use) to personal/shell_startup.py")
except Exception as e:
    warning(f"Error loading personal shell startup: {e}")

# Starts an ipython shell with access to the variables in this local scope (the imports)
IPython.start_ipython(user_ns=locals(), config=ipython_config)
