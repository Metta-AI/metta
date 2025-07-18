#!/usr/bin/env python3
"""Test script to verify BBC curriculum uses reward observations."""

from omegaconf import OmegaConf
import hydra
import os
import sys

# Add metta to path
sys.path.insert(0, os.path.abspath('.'))

# Register resolvers before using configs
from metta.common.util.resolvers import register_resolvers
register_resolvers()

# Initialize hydra with relative path
with hydra.initialize(config_path="configs", version_base=None):
    try:
        # Load the BBC basic task config directly
        cfg = hydra.compose(config_name="env/mettagrid/curriculum/bbc/tasks/basic")
        
        # Extract the actual curriculum config from the nested structure
        curriculum_cfg = cfg.env.mettagrid.curriculum.bbc.tasks
        
        print("Curriculum config:")
        print(OmegaConf.to_yaml(curriculum_cfg))
        
        # Instantiate the curriculum
        curriculum = hydra.utils.instantiate(curriculum_cfg)
        
        print(f"\nCurriculum type: {type(curriculum).__name__}")
        print(f"Use reward observations: {curriculum._use_reward_observations}")
        print(f"Reward types: {curriculum._reward_types}")
        print(f"Reward aggregation: {curriculum._reward_aggregation}")
        
        # Get a task
        task = curriculum.get_task()
        print(f"\nGot task: {task.id()}")
        
        # Simulate completing a task with reward observations
        reward_observations = {
            "ore_red": 0.5,
            "battery_red": 0.3,
            "heart": 0.8,
            "laser": 0.0,
            "armor": 0.2,
            "blueprint": 0.1,
            "total": 0.4  # Average reward
        }
        
        print(f"\nCompleting task with reward observations: {reward_observations}")
        # Get the actual task ID from the curriculum's task list
        task_id = list(curriculum._curricula.keys())[0]
        curriculum.complete_task(task_id, reward_observations)
        
        # Check if weights were updated
        print(f"\nTask weights after completion:")
        for tid, weight in curriculum._task_weights.items():
            print(f"  {tid}: {weight:.4f}")
        
        # Check if per-reward trackers exist
        if hasattr(curriculum, '_lp_trackers_per_reward'):
            print(f"\nPer-reward LP trackers created: {list(curriculum._lp_trackers_per_reward.keys())}")
        
        # Get curriculum stats
        if hasattr(curriculum, 'stats'):
            stats = curriculum.stats()
            print(f"\nCurriculum stats keys: {list(stats.keys()) if stats else 'None'}")
        
        print("\nTest completed successfully!")
        
    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()