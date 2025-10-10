import importlib


def ensure_mamba_available() -> None:
    if importlib.util.find_spec("flash_attn") is None:
        raise ModuleNotFoundError(
            "flash-attn is required on this platform. Did you activate the requirements profile?"
        )
