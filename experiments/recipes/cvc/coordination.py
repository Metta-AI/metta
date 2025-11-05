"""Coordination-focused CoGs vs Clips training entrypoint."""

from functools import partial

from experiments.recipes.cvc.core import play as _play, train_coordination as train

play = partial(_play, mission_name="go_together", num_cogs=4)
play.__doc__ = (
    "Play a coordination-heavy mission (defaults to go_together with 4 cogs)."
)

__all__ = ["train", "play"]
