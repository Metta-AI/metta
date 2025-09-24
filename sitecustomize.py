import warnings

# Suppress Gym warnings about being unmaintained
warnings.filterwarnings("ignore", message=".*Gym has been unmaintained.*", category=UserWarning)

# Torch still imports the deprecated `pynvml` shim, which emits a noisy FutureWarning.
warnings.filterwarnings(
    "ignore",
    message="The pynvml package is deprecated",
    category=FutureWarning,
)
