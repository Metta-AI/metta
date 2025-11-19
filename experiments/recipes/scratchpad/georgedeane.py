from experiments.recipes.in_context_learning import assemblers
from metta.tools.play import PlayTool
from metta.tools.replay import ReplayTool
from metta.tools.sim import SimTool
from metta.tools.train import TrainTool
from metta.sim.simulation_config import SimulationConfig
from experiments.evals.in_context_learning import assemblers as eval_assemblers
# This file is for local experimentation only. It is not checked in, and therefore won't be usable on skypilot

# You can run these functions locally with e.g. `./tools/run.py experiments.recipes.scratchpad.georgedeane.train`
# The VSCode "Run and Debug" section supports options to run these functions.


def train() -> TrainTool:
    env = arena.make_mettagrid()
    env.game.max_steps = 100
    cfg = arena.train(
        curriculum=arena.make_curriculum(env),
    )
    assert cfg.trainer.evaluation is not None
    # When we're using this file, we training locally on code that's likely not to be checked in, let alone pushed.
    # So remote evaluation probably doesn't make sense.
    cfg.trainer.evaluation.evaluate_remote = False
    cfg.trainer.evaluation.evaluate_local = True
    return cfg


# In-context learning
def play() -> PlayTool:
    env = eval_assemblers.make_assembler_eval_env(
                num_agents=1,
                max_steps=512,
                num_altars = 2,
                num_converters=0,
                width=6,
                height=6,
                altar_position=["W"],
                altar_input="one")


    return PlayTool(
        sim=SimulationConfig(
            env=env,
            name="in_context_assemblers",
        ),
    )


# def play() -> PlayTool:
#     return assemblers.play()

# Navigation evals
# def play() -> PlayTool:
#     env = navigation.make_nav_ascii_env(name = "corridors", max_steps = 100, num_agents = 1, num_instances = 4)
#     return PlayTool(
#         sim=SimulationConfig(
#             env=env,
#             name="navigation/corridors",
#         ),
#     )


def replay() -> ReplayTool:
    task_generator_cfg = icl_resource_chain.ConverterChainTaskGenerator.Config(
        chain_lengths=[6],
        num_sinks=[2],
    )
    task_generator = icl_resource_chain.ConverterChainTaskGenerator(task_generator_cfg)
    env = task_generator.get_task(0)

    return ReplayTool(
        sim=SimulationConfig(
            env=env,
            name="in_context_resource_chain",
        ),
        policy_uri="wandb://run/icl_assemblers3_two_agent_two_altars_pattern.2025-09-22",
    )


def evaluate(run: str = "local.georgedeane.1") -> SimTool:
    cfg = arena.evaluate(policy_uri=f"wandb://run/{run}")

    # If your run doesn't exist, try this:
    # cfg = arena.evaluate(policy_uri="wandb://run/daveey.combat.lpsm.8x4")
    return cfg


# curriculum_args = {
#     # 1) Single agent, only altars; positions vary (Any, W+E, N+S)
#     "single_agent_only_altars": {
#         "num_agents": [1],
#         "num_altars": [2],
#         "num_converters": [0],
#         "widths": [6, 10, 12],
#         "heights": [6, 10, 12],
#         "generator_positions": [["Any"]],
#         "altar_positions": [["Any"], ["W"], ["E"], ["N"], ["S"]],
#     },
#
#     # 2) Single agent, 1 converter + 1 altar; positions Any or single-side (N/S/E/W) TODO: cur
#     "single_agent_converter_and_altar": {
#         "num_agents": [1],
#         "num_altars": [1],
#         "num_converters": [1],
#         "widths": [6, 10, 12],
#         "heights": [6, 10, 12],
#         "generator_positions": [["Any"], ["N"], ["S"], ["E"], ["W"]],
#         "altar_positions": [["Any"], ["N"], ["S"], ["E"], ["W"]],
#         "altar_inputs": ["one"],                          # one converter available
#     },
#
#     # 3) Single agent, 2 converters + 1 altar; only one converter required
#     #    Positions: either Any for both, or both constrained to N+S or E+W error: unknown object type generator_green
#     "single_agent_two_converters_one_active": {
#         "num_agents": [1],
#         "num_altars": [1],
#         "num_converters": [2],
#         "widths": [6, 10, 12],
#         "heights": [6, 10, 12],
#         "generator_positions": [["Any"], ["N"], ["S"], ["E"], ["W"]],
#         "altar_positions": [["Any"]],
#         "altar_inputs": ["one"],                          # only one converter’s output needed
#     },
#
#     # 4) Multi-agent (up to 2 agents), 2 altars; Any positions ISSUE: cooldown needs to be longer or we'll get degenerate strategies. is there another way to enforce alternate usage?
#     "multi_agent_any": {
#         "num_agents": [1, 2],
#         "num_altars": [2],
#         "num_converters": [0],
#         "widths": [4, 6, 8, 10],
#         "heights": [4, 6, 8, 10],
#         "generator_positions": [["Any"]],          # no converters, ignored
#         "altar_positions": [["Any"]],
#         "altar_inputs": ["one"],
#     },
#
#     # 5) Multi-agent (2 agents), altars positioned N+S or W+E
#     "multi_agent_altars": {
#         "num_agents": [2],
#         "num_altars": [2],
#         "num_converters": [0],
#         "widths": [4, 6, 8, 10],
#         "heights": [4, 6, 8, 10],
#         "generator_positions": [["Any", "Any"]],          # no converters, ignored
#         "altar_positions": [["N", "S"], ["W", "E"]],
#         "altar_inputs": ["one"],
#     },
#     "multi_agent_both": {
#         "num_agents": [2],
#         "num_altars": [1],
#         "num_converters": [2],
#         "widths": [4, 6, 8, 10],
#         "heights": [4, 6, 8, 10],
#         "generator_positions": [["Any", "Any"], ["N", "S"], ["E", "W"]],
#         "altar_positions": [["Any"]],
#         "altar_inputs": ["both"],
#     },
#     "multi_agent_one_converter_one_altar": {
#         "num_agents": [2],
#         "num_altars": [1],
#         "num_converters": [1],
#         "widths": [4, 6, 8, 10],
#         "heights": [4, 6, 8, 10],
#         "generator_positions": [["Any"], ["N"], ["S"], ["E"], ["W"]],
#         "altar_positions": [["Any"], ["N"], ["S"], ["E"], ["W"]],
#         "altar_inputs": ["one"],
#     },
# }
