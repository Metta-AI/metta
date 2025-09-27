"""Minimal HuggingFace compatibility stubs for vendored Mamba."""

from typing import Any, Dict


def load_config_hf(*args, **kwargs) -> Dict[str, Any]:
    raise NotImplementedError("HuggingFace loading is not supported in the vendored Mamba build.")


def load_state_dict_hf(*args, **kwargs):
    raise NotImplementedError("HuggingFace loading is not supported in the vendored Mamba build.")
