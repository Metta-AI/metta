import warnings

from metta._cuda_env import prune_conflicting_nvidia_paths

# Suppress Gym warnings about being unmaintained
warnings.filterwarnings("ignore", message=".*Gym has been unmaintained.*", category=UserWarning)

prune_conflicting_nvidia_paths()
