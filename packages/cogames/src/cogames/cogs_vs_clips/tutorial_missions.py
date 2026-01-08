from typing import override

from cogames.cogs_vs_clips.mission import Mission, MissionVariant
from cogames.cogs_vs_clips.sites import TRAINING_FACILITY


class TutorialVariant(MissionVariant):
    name: str = "tutorial_mode"
    description: str = "High energy regen for learning."

    @override
    def modify_mission(self, mission: Mission) -> None:
        mission.energy_regen_amount = 1

    @override
    def modify_env(self, mission: Mission, env) -> None:
        env.game.max_steps = max(env.game.max_steps, 1000)


TutorialMission = Mission(
    name="tutorial",
    description="Learn the basics of CoGames: Gather, Craft, and Deposit.",
    site=TRAINING_FACILITY,
    variants=[TutorialVariant()],
)
