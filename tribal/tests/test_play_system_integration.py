#!/usr/bin/env python3
"""
Debug script to test the exact flow that the play system uses.
"""

import sys
from pathlib import Path

# Add tribal to path
sys.path.insert(0, str(Path(__file__).parent / "tribal" / "bindings" / "generated"))

from metta.cogworks.curriculum import Curriculum
from metta.rl.vecenv import make_vecenv
from metta.sim.tribal_genny import TribalEnvConfig


def test_play_flow():
    print("Creating tribal environment config...")
    tribal_config = TribalEnvConfig.with_nim_defaults(label="tribal_debug", desync_episodes=True)

    print("Creating curriculum...")
    # Create a simple curriculum like the play system does
    from experiments.recipes.tribal_basic import tribal_env_curriculum

    curriculum_config = tribal_env_curriculum(tribal_config)
    curriculum = Curriculum(curriculum_config)

    print("Creating vectorized environment...")
    vecenv = make_vecenv(
        curriculum=curriculum,
        vectorization="serial",
        num_envs=1,
        num_workers=1,
        render_mode="rgb_array",
        is_training=False,
    )

    print("Getting wrapped environment...")
    env = vecenv.envs[0]  # This is what sim.get_env() returns

    print("Testing wrapped environment attributes...")
    try:
        print(f"action_names: {env.action_names}")
    except Exception as e:
        print(f"ERROR accessing action_names through wrapper: {e}")
        import traceback

        traceback.print_exc()

    try:
        print(f"max_steps: {env.max_steps}")
    except Exception as e:
        print(f"ERROR accessing max_steps through wrapper: {e}")
        import traceback

        traceback.print_exc()

    try:
        print(f"height: {env.height}")
    except Exception as e:
        print(f"ERROR accessing height through wrapper: {e}")
        import traceback

        traceback.print_exc()

    try:
        print("Creating replay dict...")
        replay_dict = {
            "version": 2,
            "action_names": env.action_names,
            "item_names": getattr(env, "resource_names", []),
            "type_names": getattr(env, "object_type_names", []),
            "num_agents": env.num_agents,
            "max_steps": env.max_steps,
            "map_size": [env.height, env.width],
            "file_name": "live_play",
            "steps": [],
        }
        print(f"Replay dict created successfully: {len(replay_dict)} keys")
    except Exception as e:
        print(f"ERROR creating replay dict: {e}")
        import traceback

        traceback.print_exc()

    print("Play flow test completed!")


if __name__ == "__main__":
    test_play_flow()
