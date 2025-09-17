import warnings

# Suppress Gym warnings about being unmaintained
warnings.filterwarnings("ignore", message=".*Gym has been unmaintained.*", category=UserWarning)
