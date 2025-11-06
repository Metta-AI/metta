import importlib.util
import sys

import IPython
import traitlets.config

import metta.common.util.fs
import metta.setup.utils

__name__ = "__ipython__"

REPO_ROOT = metta.common.util.fs.get_repo_root()
CONFIGS_DIR = REPO_ROOT / "configs"


def help_configs() -> None:
    metta.setup.utils.header("=== Usage Examples ===")
    metta.setup.utils.success("# Load configs:")
    metta.setup.utils.info('cfg = load_cfg("sim_job.yaml")')
    metta.setup.utils.info('cfg = load_cfg("trainer/trainer.yaml")')
    metta.setup.utils.success("# Load configs with overrides:")
    metta.setup.utils.info(
        'cfg = load_cfg("train_job.yaml", ["training_env.curriculum=/env/mettagrid/arena/advanced"])'
    )
    metta.setup.utils.success("# Load checkpoints:")
    metta.setup.utils.info(
        'artifact = CheckpointManager.load_artifact_from_uri("file://./train_dir/my_run/checkpoints/my_run:v12.mpt")'
    )
    metta.setup.utils.info(
        'artifact = CheckpointManager.load_artifact_from_uri("s3://bucket/path/my_run/checkpoints/my_run:v12.mpt")'
    )
    metta.setup.utils.info('policy = artifact.policy  # or artifact.instantiate(game_rules, torch.device("cpu"))')
    metta.setup.utils.success("# Create checkpoint manager:")
    metta.setup.utils.info('cm = CheckpointManager(run="my_run", run_dir="./train_dir")')


# Create a new IPython config object
ipython_config = traitlets.config.Config()
ipython_config.InteractiveShellApp.extensions = ["autoreload"]
ipython_config.InteractiveShellApp.exec_lines = [
    "%autoreload 2",
    "success('Autoreload enabled: modules will be reloaded automatically when changed.')",
    "info('Use help_configs() to see usage examples')",
    "info('CheckpointManager is available for loading/saving checkpoints')",
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

        metta.setup.utils.success("Personal shell startup loaded successfully")
    except Exception as e:
        metta.setup.utils.warning(f"Error loading personal shell startup: {e}")
else:
    metta.setup.utils.info(
        "You can add custom shell startup code (e.g. importing functions you often use) to personal/shell_startup.py"
    )

# Starts an ipython shell with access to the variables in this local scope (the imports)
IPython.start_ipython(user_ns=locals(), config=ipython_config)
