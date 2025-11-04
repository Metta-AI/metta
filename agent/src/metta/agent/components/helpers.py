from importlib import util


def ensure_mamba_available() -> None:
    if util.find_spec("flash_attn") is None:
        raise ModuleNotFoundError("flash-attn is required on this platform. Did you activate the requirements profile?")
