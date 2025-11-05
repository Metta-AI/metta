"""Cogs vs Clips (CVC) training recipes."""

from experiments.recipes.cvc.coordination import train as train_coordination
from experiments.recipes.cvc.curriculum import (
    make_curriculum,
    make_training_env,
    train,
)
from experiments.recipes.cvc.evaluation import evaluate, make_eval_suite
from experiments.recipes.cvc.experiment import experiment
from experiments.recipes.cvc.large_maps import train as train_large_maps
from experiments.recipes.cvc.medium_maps import train as train_medium_maps
from experiments.recipes.cvc.play import play, play_training_env
from experiments.recipes.cvc.single_mission import train as train_single_mission
from experiments.recipes.cvc.small_maps import train as train_small_maps

__all__ = [
    "evaluate",
    "experiment",
    "make_curriculum",
    "make_eval_suite",
    "make_training_env",
    "play",
    "play_training_env",
    "train",
    "train_coordination",
    "train_large_maps",
    "train_medium_maps",
    "train_single_mission",
    "train_small_maps",
]
