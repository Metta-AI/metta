"""Curriculum for Machina missions.

Progresses from easier scattered resource missions to the full open world.
"""

from __future__ import annotations

from typing import Optional, Sequence

from metta.cogworks.curriculum.curriculum import CurriculumConfig
from metta.tools.train import TrainTool
from recipes.experiment import cogs_v_clips

MACHINA_CURRICULUM_MISSIONS: tuple[str, ...] = (
    "training_facility.harvest",
    "hello_world.easy_hearts_hello_world",
    "machina_scattered_easy",
    "machina_scattered_medium",
    "machina_1.open_world",
)


def train(
    num_cogs: int = 4,
    curriculum: Optional[CurriculumConfig] = None,
    base_missions: Optional[list[str]] = None,
    enable_detailed_slice_logging: bool = False,
    variants: Optional[Sequence[str]] = None,
    eval_variants: Optional[Sequence[str]] = None,
    eval_difficulty: str | None = "standard",
    mission: str | None = None,
) -> TrainTool:
    """Train on the Machina curriculum."""
    if base_missions is None:
        base_missions = list(MACHINA_CURRICULUM_MISSIONS)

    return cogs_v_clips.train(
        num_cogs=num_cogs,
        curriculum=curriculum,
        base_missions=base_missions,
        enable_detailed_slice_logging=enable_detailed_slice_logging,
        variants=variants,
        eval_variants=eval_variants,
        eval_difficulty=eval_difficulty,
        mission=mission,
    )

