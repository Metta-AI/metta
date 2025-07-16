import importlib.util
import logging
import sys
from functools import partial

import IPython
from traitlets.config import Config as IPythonConfig

from metta.common.util.fs import get_repo_root
from metta.setup.utils import header, info, success, warning

sys.path.insert(0, str(get_repo_root() / "tools"))
from validate_config import load_and_print_config  # type: ignore

__name__ = "__ipython__"
logger = logging.getLogger(__name__)

REPO_ROOT = get_repo_root()
CONFIGS_DIR = REPO_ROOT / "configs"

load_cfg = partial(load_and_print_config, exit_on_failure=False, print_cfg=False)

from metta.common.util.stats_client_cfg import get_stats_client  # noqa
from metta.agent.policy_store import PolicyStore  # noqa
from metta.app_backend.stats_client import StatsClient  # noqa


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


personal_dir = REPO_ROOT / "personal"
personal_startup_script = personal_dir / "shell_startup.py"
if personal_startup_script.exists():
    try:
        if str(personal_dir) not in sys.path:
            sys.path.insert(0, str(personal_dir))
        spec = importlib.util.spec_from_file_location("personal.shell_startup", personal_startup_script)
        if spec and spec.loader:
            shell_startup = importlib.util.module_from_spec(spec)
            # Inject our current globals into the module's namespace
            current_globals = globals().copy()
            current_globals.update(locals())
            for key, value in current_globals.items():
                if key not in ["__name__", "__file__", "__cached__"]:
                    setattr(shell_startup, key, value)

            # Now execute the module with our globals available
            spec.loader.exec_module(shell_startup)
            # Import names from shell_startup into our namespace
            for name in dir(shell_startup):
                locals()[name] = getattr(shell_startup, name)

        success("Personal shell startup loaded successfully")
    except Exception as e:
        warning(f"Error loading personal shell startup: {e}")
else:
    info("You can add custom shell startup code (e.g. importing functions you often use) to personal/shell_startup.py")

# Starts an ipython shell with access to the variables in this local scope (the imports)
IPython.start_ipython(user_ns=locals(), config=ipython_config)
