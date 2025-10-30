"""Test adaptive exploration calculation."""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent / "packages" / "cogames" / "src"))

from cogames.cogs_vs_clips.difficulty_variants import EASY, apply_difficulty
from cogames.cogs_vs_clips.exploration_experiments import Experiment1Mission, Experiment2Mission
from cogames.policy.scripted_agent import MettaGridEnv, ScriptedAgentPolicy, Hyperparameters

for exp_name, mission_class in [("EXP1-EASY", Experiment1Mission), ("EXP2-EASY", Experiment2Mission)]:
    mission = mission_class()
    apply_difficulty(mission, EASY)
    mission = mission.instantiate(mission.site.map_builder, num_cogs=1)
    env_config = mission.make_env()
    env = MettaGridEnv(env_config)

    hyperparams = Hyperparameters(strategy_type="explorer_first", exploration_phase_steps=100, min_energy_for_silicon=70)
    policy = ScriptedAgentPolicy(env, hyperparams=hyperparams)

    impl = policy._impl
    print(f"{exp_name}:")
    print(f"  Map: {impl._map_height}x{impl._map_width} = {impl._map_height * impl._map_width} cells")
    print(f"  Base exploration: {hyperparams.exploration_phase_steps}")
    print(f"  Adaptive exploration: {impl._adaptive_exploration_steps}")
    print()

