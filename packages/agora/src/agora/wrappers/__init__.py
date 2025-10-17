"""Environment wrappers for curriculum learning."""

# Optional pufferlib integration
try:
    from agora.wrappers.puffer import CurriculumEnv

    __all__ = ["CurriculumEnv"]
except ImportError:
    # pufferlib not available
    __all__ = []  # type: ignore[misc]
