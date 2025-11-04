"""
Hyperparameter Presets for Scripted Agent

Four distinct preset behaviours to cover exploration, courier routing, hoarding, and story mode.
"""

from cogames.policy.scripted_agent.hyperparameters import (
    courier_params,
    hoarder_params,
    minimal_baseline_params,
    scout_params,
    story_mode_params,
)

STORY_MODE = story_mode_params()
COURIER = courier_params()
SCOUT = scout_params()
HOARDER = hoarder_params()
MINIMAL_BASELINE = minimal_baseline_params()

HYPERPARAMETER_PRESETS = {
    "story_mode": STORY_MODE,
    "courier": COURIER,
    "scout": SCOUT,
    "hoarder": HOARDER,
    "minimal_baseline": MINIMAL_BASELINE,
    # Legacy aliases
    "balanced": COURIER,
    "explorer_long": SCOUT,
    "greedy_conservative": HOARDER,
    "efficiency_heavy": HOARDER,
    "sequential_baseline": STORY_MODE,
}
