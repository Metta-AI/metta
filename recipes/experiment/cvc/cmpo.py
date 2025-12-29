"""CMPO CoGs vs Clips training entry points."""

from typing import Optional, Sequence

from metta.cogworks.curriculum.curriculum import CurriculumConfig
from metta.rl.training.teacher import TeacherConfig
from metta.tools.train import TrainTool
from recipes.experiment import cogs_v_clips
from recipes.experiment.losses.cmpo import cmpo_losses


def train(
    num_cogs: int = 4,
    curriculum: Optional[CurriculumConfig] = None,
    mission: Optional[str] = None,
    base_missions: Optional[list[str]] = None,
    enable_detailed_slice_logging: bool = False,
    variants: Optional[Sequence[str]] = None,
    eval_variants: Optional[Sequence[str]] = None,
    eval_difficulty: str | None = "standard",
    max_evals: Optional[int] = None,
    teacher: TeacherConfig | None = None,
    use_lp: bool = True,
    dr_variants: int = 0,
    dr_rewards: bool = True,
    dr_misc: bool = False,
    maps_cache_size: Optional[int] = 30,
) -> TrainTool:
    tool = cogs_v_clips.train(
        num_cogs=num_cogs,
        curriculum=curriculum,
        mission=mission,
        base_missions=base_missions,
        enable_detailed_slice_logging=enable_detailed_slice_logging,
        variants=variants,
        eval_variants=eval_variants,
        eval_difficulty=eval_difficulty,
        max_evals=max_evals,
        teacher=teacher,
        use_lp=use_lp,
        dr_variants=dr_variants,
        dr_rewards=dr_rewards,
        dr_misc=dr_misc,
        maps_cache_size=maps_cache_size,
    )
    tool.trainer.losses = cmpo_losses()
    return tool


__all__ = ["train"]
