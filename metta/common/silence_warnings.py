"""Silence deprecation warnings from Gym."""

import logging
import warnings

# Silence the deprecation warning
warnings.filterwarnings("ignore", category=DeprecationWarning)

# Silence the gym logger message about upgrading
logging.getLogger("gym.error").setLevel(logging.ERROR)
