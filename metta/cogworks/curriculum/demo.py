import metta.cogworks.curriculum as cc
import mettagrid.builder.envs as eb
from metta.cogworks.curriculum.curriculum import Curriculum, CurriculumConfig
from metta.cogworks.curriculum.task_generator import Span

arena = eb.make_arena(num_agents=24)

# disable swap
arena.game.actions.swap.enabled = False

# make a set of training tasks for the arena
arena_tasks = cc.bucketed(arena)

# arena_tasks.add_bucket("game.level_map.num_agents", [1, 2, 3, 4, 6, 24])

# arena_tasks.add_bucket("game.level_map.width", [10, 20, 30, 40, 50])
# arena_tasks.add_bucket("game.level_map.height", [10, 20, 30, 40, 50])

for item in arena.game.resource_names:
    arena_tasks.add_bucket(f"game.agent.rewards.inventory.{item}", [0, Span(0, 1.0)])
    arena_tasks.add_bucket(f"game.agent.rewards.inventory_max.{item}", [1, 2])

# enable or disable attacks. we use cost instead of 'enabled'
# to maintain action space consistency.
arena_tasks.add_bucket("game.actions.attack.consumed_resources.laser", [1, 100])

curriculum_cfg = CurriculumConfig(task_generator=arena_tasks)

print(curriculum_cfg.model_dump_json(indent=2))
print(curriculum_cfg.task_generator.model_dump_json(indent=2))
curriculum = Curriculum(curriculum_cfg)

# print("Generating 10 tasks")
# for _ in range(10):
#     task = curriculum.get_task()
#     print("================ TASK ==================")
#     print(task._task_id, json.dumps(task.get_env_cfg().model_dump(), indent=2))
