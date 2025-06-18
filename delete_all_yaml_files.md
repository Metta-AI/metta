# Replace YAML with Python

First we need to consider all of the YAML files. There are a lot of them.
Some of them are configuration files and many of them define a game.
By volume, game definition files are the bulk of the files.


## Current YAML files:

### Defines run and other basic stuff
configs/common.yaml

### Defines logging and other basic stuff
configs/hydra.yaml

### Job configs:
configs/play_job.yaml
configs/train_job.yaml
configs/renderer_job.yaml
configs/replay_job.yaml
configs/analyze_job.yaml
configs/sim_job.yaml
configs/dashboard_job.yaml
configs/sweep_job.yaml

### Config to train on different hardware:
configs/hardware/mac_serial.yaml
configs/hardware/pufferbox.yaml
configs/hardware/macbook.yaml
configs/hardware/mac_parallel.yaml
configs/hardware/aws.yaml
configs/hardware/github.yaml

### Trainer configs related to hardware:
configs/trainer/simple.yaml
configs/trainer/simple.medium.yaml
configs/trainer/puffer.yaml
configs/trainer/trainer.yaml
configs/trainer/g16.yaml

### User configs:
configs/user/alex.yaml
configs/user/arikalinowski.yaml
configs/user/cursor.yaml
configs/user/daveey.yaml
configs/user/daphne.yaml
configs/user/rwalters.yaml
configs/user/relh.yaml
configs/user/gabrielk.yaml
configs/user/sasmith.yaml
configs/user/jack.yaml
configs/user/berekuk.yaml
configs/user/lars.yaml
configs/user/ptsier.yaml
configs/user/georgedeane.yaml
configs/user/me.yaml

### Agent configs (neural network architectures)
configs/agent/latent_fourier.yaml
configs/agent/simple.yaml
configs/agent/latent_attn.yaml
configs/agent/simple_multi_attn.yaml
configs/agent/simple_NAS008.yaml
configs/agent/latent_rope.yaml
configs/agent/self_attn.yaml
configs/agent/robust_cross.yaml
configs/agent/reference_design.yaml
configs/agent/latent_cat.yaml
configs/agent/simple_token_to_box.yaml

### Sets up the basic metta grid game:
configs/env/mettagrid/mettagrid.yaml
configs/env/mettagrid/simple.yaml

### This all the of games that we can play:
configs/env/mettagrid/multiagent/multiagent/boxshare.yaml
configs/env/mettagrid/multiagent/experiments/varied_terrain_dense.yaml
configs/env/mettagrid/multiagent/experiments/defaults.yaml
configs/env/mettagrid/multiagent/experiments/boxshare.yaml
configs/env/mettagrid/multiagent/experiments/terrain_from_numpy.yaml
configs/env/mettagrid/multiagent/experiments/narrow_world.yaml
configs/env/mettagrid/multiagent/experiments/varied_terrain_sparse.yaml
configs/env/mettagrid/multiagent/experiments/varied_terrain.yaml
configs/env/mettagrid/multiagent/experiments/varied_terrain_balanced.yaml
configs/env/mettagrid/multiagent/experiments/varied_terrain_maze.yaml
configs/env/mettagrid/multiagent/experiments/manhatten.yaml
configs/env/mettagrid/multiagent/experiments/cylinder_world.yaml
configs/env/mettagrid/multiagent/experiments/boxy.yaml
configs/env/mettagrid/memory/evals/corners.yaml
configs/env/mettagrid/memory/evals/defaults.yaml
configs/env/mettagrid/memory/evals/tease_small.yaml
configs/env/mettagrid/memory/evals/which_way.yaml
configs/env/mettagrid/memory/evals/little_landmark_hard.yaml
configs/env/mettagrid/memory/evals/little_landmark_easy.yaml
configs/env/mettagrid/memory/evals/medium_sequence.yaml
configs/env/mettagrid/memory/evals/access_cross.yaml
configs/env/mettagrid/memory/evals/hall_of_mirrors.yaml
configs/env/mettagrid/memory/evals/medium.yaml
configs/env/mettagrid/memory/evals/passing_things.yaml
configs/env/mettagrid/memory/evals/hard_sequence.yaml
configs/env/mettagrid/memory/evals/spacey_memory.yaml
configs/env/mettagrid/memory/evals/choose_wisely.yaml
configs/env/mettagrid/memory/evals/memory_swirls_hard.yaml
configs/env/mettagrid/memory/evals/boxout.yaml
configs/env/mettagrid/memory/evals/easy_sequence.yaml
configs/env/mettagrid/memory/evals/lobster_legs_cues.yaml
configs/env/mettagrid/memory/evals/you_shall_not_pass.yaml
configs/env/mettagrid/memory/evals/hard.yaml
configs/env/mettagrid/memory/evals/memory_swirls.yaml
configs/env/mettagrid/memory/evals/easy.yaml
configs/env/mettagrid/memory/evals/venture_out.yaml
configs/env/mettagrid/memory/evals/tease.yaml
configs/env/mettagrid/memory/evals/journey_home.yaml
configs/env/mettagrid/memory/evals/lobster_legs.yaml
configs/env/mettagrid/memory/training/medium.yaml
configs/env/mettagrid/memory/training/hard.yaml
configs/env/mettagrid/memory/training/easy.yaml
configs/env/mettagrid/prog_3.yaml
configs/env/mettagrid/cooperation/experimental/central_table_layout.yaml
configs/env/mettagrid/cooperation/experimental/two_rooms_coord.yaml
configs/env/mettagrid/cooperation/experimental/confined_room_coord.yaml
configs/env/mettagrid/species.yaml
configs/env/mettagrid/curriculum/overcooked.yaml
configs/env/mettagrid/curriculum/simple.yaml
configs/env/mettagrid/curriculum/navigation.yaml
configs/env/mettagrid/curriculum/multiagent.yaml
configs/env/mettagrid/curriculum/navsequence.yaml
configs/env/mettagrid/curriculum/object_use.yaml
configs/env/mettagrid/curriculum/all.yaml
configs/env/mettagrid/terrain_from_numpy.yaml
configs/env/mettagrid/teams.yaml
configs/env/mettagrid/reward_dr.yaml
configs/env/mettagrid/navigation/evals/walls_outofsight.yaml
configs/env/mettagrid/navigation/evals/defaults.yaml
configs/env/mettagrid/navigation/evals/labyrinth.yaml
configs/env/mettagrid/navigation/evals/walls_sparse.yaml
configs/env/mettagrid/navigation/evals/swirls.yaml
configs/env/mettagrid/navigation/evals/cylinder_easy.yaml
configs/env/mettagrid/navigation/evals/emptyspace_sparse.yaml
configs/env/mettagrid/navigation/evals/obstacles0.yaml
configs/env/mettagrid/navigation/evals/radial_mini.yaml
configs/env/mettagrid/navigation/evals/cylinder.yaml
configs/env/mettagrid/navigation/evals/obstacles1.yaml
configs/env/mettagrid/navigation/evals/walkaround.yaml
configs/env/mettagrid/navigation/evals/honeypot.yaml
configs/env/mettagrid/navigation/evals/obstacles2.yaml
configs/env/mettagrid/navigation/evals/walls_withinsight.yaml
configs/env/mettagrid/navigation/evals/knotty.yaml
configs/env/mettagrid/navigation/evals/radialmaze.yaml
configs/env/mettagrid/navigation/evals/thecube.yaml
configs/env/mettagrid/navigation/evals/obstacles3.yaml
configs/env/mettagrid/navigation/evals/radial_large.yaml
configs/env/mettagrid/navigation/evals/corridors.yaml
configs/env/mettagrid/navigation/evals/memory_palace.yaml
configs/env/mettagrid/navigation/evals/radial_small.yaml
configs/env/mettagrid/navigation/evals/emptyspace_withinsight.yaml
configs/env/mettagrid/navigation/evals/emptyspace_outofsight.yaml
configs/env/mettagrid/navigation/evals/wanderout.yaml
configs/env/mettagrid/navigation/training/varied_terrain_dense.yaml
configs/env/mettagrid/navigation/training/terrain_from_numpy.yaml
configs/env/mettagrid/navigation/training/varied_terrain_sparse.yaml
configs/env/mettagrid/navigation/training/varied_terrain_balanced.yaml
configs/env/mettagrid/navigation/training/varied_terrain_maze.yaml
configs/env/mettagrid/navigation/training/cylinder_world.yaml
configs/env/mettagrid/debug.yaml
configs/env/mettagrid/laser_tag.yaml
configs/env/mettagrid/game/map_builder/terrain.yaml
configs/env/mettagrid/game/map_builder/mapgen_species.yaml
configs/env/mettagrid/game/map_builder/simple.yaml
configs/env/mettagrid/game/map_builder/mapgen_simple.yaml
configs/env/mettagrid/game/map_builder/species.yaml
configs/env/mettagrid/game/map_builder/mapgen_convchain.yaml
configs/env/mettagrid/game/map_builder/mapgen_bsp_plus_connected.yaml
configs/env/mettagrid/game/map_builder/mapgen_maze_basic.yaml
configs/env/mettagrid/game/map_builder/mapgen_demo.yaml
configs/env/mettagrid/game/map_builder/mapgen_wfc_demo.yaml
configs/env/mettagrid/game/map_builder/mapgen_bsp_complex.yaml
configs/env/mettagrid/game/map_builder/mapgen_maze.yaml
configs/env/mettagrid/game/map_builder/mapgen_bsp.yaml
configs/env/mettagrid/game/map_builder/mapgen_wfc_simple.yaml
configs/env/mettagrid/game/map_builder/maze.yaml
configs/env/mettagrid/game/map_builder/mapgen_auto.yaml
configs/env/mettagrid/game/map_builder/random_scene.yaml
configs/env/mettagrid/game/map_builder/load_random.yaml
configs/env/mettagrid/game/map_builder/load.yaml
configs/env/mettagrid/puffer.yaml
configs/env/mettagrid/object_use/evals/defaults.yaml
configs/env/mettagrid/object_use/evals/generator_use.yaml
configs/env/mettagrid/object_use/evals/temple_use_free.yaml
configs/env/mettagrid/object_use/evals/swap_in.yaml
configs/env/mettagrid/object_use/evals/armory_use.yaml
configs/env/mettagrid/object_use/evals/armory_use_free.yaml
configs/env/mettagrid/object_use/evals/generator_use_free.yaml
configs/env/mettagrid/object_use/evals/mine_use.yaml
configs/env/mettagrid/object_use/evals/shoot_out.yaml
configs/env/mettagrid/object_use/evals/lasery_use_free.yaml
configs/env/mettagrid/object_use/evals/full_sequence.yaml
configs/env/mettagrid/object_use/evals/swap_out.yaml
configs/env/mettagrid/object_use/evals/lasery_use.yaml
configs/env/mettagrid/object_use/evals/altar_use_free.yaml
configs/env/mettagrid/object_use/training/easy_all_objects.yaml
configs/env/mettagrid/bases.yaml
configs/env/mettagrid/navigation_sequence/evals/defaults.yaml
configs/env/mettagrid/navigation_sequence/evals/swirls.yaml
configs/env/mettagrid/navigation_sequence/evals/cylinder_easy.yaml
configs/env/mettagrid/navigation_sequence/evals/obstacles0.yaml
configs/env/mettagrid/navigation_sequence/evals/radial_mini.yaml
configs/env/mettagrid/navigation_sequence/evals/cylinder.yaml
configs/env/mettagrid/navigation_sequence/evals/obstacles1.yaml
configs/env/mettagrid/navigation_sequence/evals/walkaround.yaml
configs/env/mettagrid/navigation_sequence/evals/honeypot.yaml
configs/env/mettagrid/navigation_sequence/evals/obstacles2.yaml
configs/env/mettagrid/navigation_sequence/evals/knotty.yaml
configs/env/mettagrid/navigation_sequence/evals/thecube.yaml
configs/env/mettagrid/navigation_sequence/evals/obstacles3.yaml
configs/env/mettagrid/navigation_sequence/evals/radial_large.yaml
configs/env/mettagrid/navigation_sequence/evals/corridors.yaml
configs/env/mettagrid/navigation_sequence/evals/memory_palace.yaml
configs/env/mettagrid/navigation_sequence/evals/radial_small.yaml
configs/env/mettagrid/navigation_sequence/experiments/mem_maze.yaml
configs/env/mettagrid/navigation_sequence/experiments/sequence_dense.yaml
configs/env/mettagrid/navigation_sequence/experiments/sequence_defaults.yaml
configs/env/mettagrid/navigation_sequence/experiments/mem_defaults.yaml
configs/env/mettagrid/navigation_sequence/experiments/mem_dense.yaml
configs/env/mettagrid/navigation_sequence/experiments/mem_sparse.yaml
configs/env/mettagrid/navigation_sequence/experiments/terrain_from_numpy.yaml
configs/env/mettagrid/navigation_sequence/experiments/triplets_balanced.yaml
configs/env/mettagrid/navigation_sequence/experiments/sequence_maze.yaml
configs/env/mettagrid/navigation_sequence/experiments/mem_balanced.yaml
configs/env/mettagrid/navigation_sequence/experiments/sequence_sparse.yaml
configs/env/mettagrid/navigation_sequence/experiments/triplets_dense.yaml
configs/env/mettagrid/navigation_sequence/experiments/sequence_balanced.yaml
configs/env/mettagrid/navigation_sequence/experiments/triplets_sparse.yaml
configs/env/mettagrid/navigation_sequence/experiments/triplets_defaults.yaml
configs/env/mettagrid/navigation_sequence/experiments/triplets_maze.yaml
configs/env/mettagrid/navigation_sequence/experiments/cylinder_world.yaml
configs/env/mettagrid/diversity/multi_left_or_right.yaml
configs/env/mettagrid/ants.yaml
configs/env/mettagrid/spatial_memory_evals/dense_position.yaml
configs/env/mettagrid/mapgen_auto.yaml
configs/env/mettagrid/split.yaml
configs/env/mettagrid/school.yaml

### Configs for different types of sweeps:
configs/sweep/empty.yaml
configs/sweep/full.yaml
configs/sweep/pong.yaml
configs/sweep/fast.yaml
configs/sweep/cogeval_sweep.yaml

### Configs for different types of simulations (usually testing?)
configs/sim/simple.yaml
configs/sim/navigation.yaml
configs/sim/nav_sequence.yaml
configs/sim/sweep_eval.yaml
configs/sim/memory.yaml
configs/sim/object_use.yaml
configs/sim/sim_single.yaml
configs/sim/smoke_test.yaml
configs/sim/sim.yaml
configs/sim/all.yaml
configs/sim/sim_suite.yaml
configs/sim/all_tokenized.yaml

### Used for wandb:
configs/wandb/off.yaml
configs/wandb/metta_research.yaml

# Proposals

I have considers 4 different options for python config files:

## Python based dictionary config

Here we create a big dictionary that is then modified in python.
This is very similar to what YAML files produce and that's what our objects expect.

```python
cfg = {
  "cmd": "train",
  "data_dir": "./train_dir",
  "seed": null,
  "agent": {
    "target": "metta.agent.metta_agent.MettaAgent",
    "analyze_weights_interval": 300,
    "clip_range": 0,
    "components": [
      ...
    ]
  },
  "sim": {
    "max_time_s": 60,
    "name": "all",
    "num_episodes": 1,
    "simulations": [
      ...
    ]
  },
}
```

## Python based object config

In this mode you have an object-dictionary you modify.
This is the most pythonic option and the most flexible.

```python
cfg.cmd = "train"
cfg.data_dir = "./train_dir"
cfg.seed = None
cfg.agent.target = "metta.agent.metta_agent.MettaAgent"
cfg.agent.analyze_weights_interval = 300
cfg.agent.clip_range = 0
cfg.agent.components = [
  ...
]
cfg.sim.max_time_s = 60
cfg.sim.name = "all"
cfg.sim.num_episodes = 1
cfg.sim.simulations = [
  ...
]
```

## Python Config Objects:

Have each major object have a config object that is created first.
This is a departure from the ones above because you are creating real python objects.
These objects can have code that will get run as well.

```python
class AgentConfig(config.AgentConfig):
  target: metta.agent.metta_agent.MettaAgent
  analyze_weights_interval: 300
  clip_range: 0

  components: [
    ...
  ]

class SimConfig(config.SimConfig):
  max_time_s: 60
  name: "all"
  num_episodes: 1
  simulations: [
    ...
  ]

class MettaConfig(config.MettaConfig):
  cmd: "train"
  data_dir: "./train_dir"
  seed: None
  agent: AgentConfig
  sim: SimConfig
  ...
```

## Creates the objects directly

Then finally my favorite option, where you create the objects directly.
This is the most flexible option and the most pythonic.

```python
agent = metta.agent.metta_agent.MettaAgent(
analyze_weights_interval=300,
clip_range=0,
components=[
  ...
],
)

sim = Sim(
  max_time_s=60,
  name="all",
  num_episodes=1,
  simulations={
    ...
  }
)

metta = Metta(
  cmd="train",
  data_dir="./train_dir",
  seed=None,
  agent=agent,
  sim=sim,
  ...
)

# I would also propose training loop to exist inside your own code,
# its there to be messed with.
while metta.training():
  metta.train()
```

Hopefully this gives you a sense of the different options and things to think about.
