from typing import Optional, Sequence

from metta.agent.policies.agalite import AGaLiTeConfig
from metta.agent.policies.fast import FastConfig
from metta.agent.policies.fast_dynamics import FastDynamicsConfig
from metta.agent.policies.memory_free import MemoryFreeConfig
from metta.agent.policies.puffer import PufferPolicyConfig
from metta.agent.policies.trxl import TRXLConfig
from metta.agent.policies.vit import ViTDefaultConfig
from metta.cogworks.curriculum.curriculum import CurriculumConfig
from metta.rl.trainer_config import TorchProfilerConfig
from metta.sim.simulation_config import SimulationConfig
from metta.sweep.core import SweepParameters as SP
from metta.sweep.core import grid_search
from metta.tools.eval import EvaluateTool
from metta.tools.sweep import SweepTool
from metta.tools.train import TrainTool
from recipes.experiment.arena import BASELINE as ARENA_BASELINE
from recipes.experiment.arena import make_curriculum, mettagrid, simulations

# Architecture configurations for benchmark testing
ARCHITECTURES = {
    "vit": ViTDefaultConfig(),
    # Cortexified transformer policies
    # "transformer" and "vit_sliding" removed in favor of Cortex-based TRXL
    "fast": FastConfig(),
    "fast_dynamics": FastDynamicsConfig(),
    "memory_free": MemoryFreeConfig(),
    "agalite": AGaLiTeConfig(),
    "trxl": TRXLConfig(),
    "puffer": PufferPolicyConfig(),
}


def train(
    curriculum: Optional[CurriculumConfig] = None,
    enable_detailed_slice_logging: bool = False,
    arch_type: str = "fast",
    baseline: Optional[TrainTool] = None,
) -> TrainTool:
    if baseline is None:
        baseline = ARENA_BASELINE.model_copy(deep=True)
    else:
        baseline = baseline.model_copy(deep=True)

    curriculum = curriculum or make_curriculum(enable_detailed_slice_logging=enable_detailed_slice_logging)
    baseline.training_env.curriculum = curriculum

    eval_simulations = simulations()
    baseline.evaluator.simulations = eval_simulations

    policy_architecture = ARCHITECTURES[arch_type]
    baseline.policy_architecture = policy_architecture
    baseline.torch_profiler = TorchProfilerConfig()

    return baseline


def evaluate(policy_uris: Optional[Sequence[str]] = None) -> EvaluateTool:
    """Evaluate policies on arena simulations."""
    return EvaluateTool(simulations=simulations(), policy_uris=policy_uris or [])


def evaluate_in_sweep(policy_uri: str) -> EvaluateTool:
    """Evaluation tool for sweep runs.

    Uses 10 episodes per simulation with a 4-minute time limit to get
    reliable results quickly during hyperparameter sweeps.
    NB: Please note that this function takes a **single** policy_uri. This is the expected signature in our sweeps.
    Additional arguments are supported through eval_overrides.
    """

    # Create sweep-optimized versions of the standard evaluations
    # Use a dedicated suite name to control the metric namespace in WandB
    basic_env = mettagrid()
    basic_env.game.actions.attack.consumed_resources["laser"] = 100

    combat_env = basic_env.model_copy()
    combat_env.game.actions.attack.consumed_resources["laser"] = 1

    simulations = [
        SimulationConfig(
            suite="sweep",
            name="basic",
            env=basic_env,
            num_episodes=10,  # 10 episodes for statistical reliability
            max_time_s=240,  # 4 minutes max per simulation
        ),
        SimulationConfig(
            suite="sweep",
            name="combat",
            env=combat_env,
            num_episodes=10,
            max_time_s=240,
        ),
    ]

    return EvaluateTool(
        simulations=simulations,
        policy_uris=[policy_uri],
    )


def sweep_architecture(sweep_name: str) -> SweepTool:
    # NB: arch_type matches the corresponding input to "train", the train_entrypoint.
    architecture_parameter = SP.categorical("arch_type", list(ARCHITECTURES.keys()))
    return grid_search(
        name=sweep_name,
        recipe="recipes.experiment.simple_architecture_search.basic",
        train_entrypoint="train",
        eval_entrypoint="evaluate_in_sweep",
        objective="evaluator/eval_sweep/score",
        parameters=[architecture_parameter],
        max_trials=200,
        num_parallel_trials=8,
    )
