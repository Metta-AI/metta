"""Metta package initialization."""

from pkgutil import extend_path

from ._cuda_env import prune_conflicting_nvidia_paths

prune_conflicting_nvidia_paths()

__path__ = extend_path(__path__, __name__)
