#!/usr/bin/env python3
"""Quick simulation to see carbon collection."""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'packages', 'cogames', 'src'))

from cogames.cogs_vs_clips.eval_missions import ClipOxygen
from cogames.cogs_vs_clips.missions import get_map
from cogames.policy.scripted_agent import ScriptedAgentPolicy
from mettagrid import MettaGridEnv

def run_quick_simulation():
    """Run simulation for first 20 steps."""

    # Create the mission
    mission = ClipOxygen()
    map_builder = get_map("eval_clip_oxygen.map")
    mission_instance = mission.instantiate(
        map_builder=map_builder,
        num_cogs=1,
        variant=None
    )

    # Create environment
    env_cfg = mission_instance.make_env()
    env = MettaGridEnv(env_cfg)

    # Create agent
    policy = ScriptedAgentPolicy(env=env)
    agent = policy.agent_policy(agent_id=0)

    # Access the internal implementation for debugging
    impl = agent._base_policy

    # Run simulation with comprehensive logging
    obs = env.reset()
    step_count = 0
    max_steps = 20

    print("=== QUICK SIMULATION LOG ===")
    print(f"Max steps: {max_steps}\n")

    while step_count < max_steps:
        # Get action
        action = agent.step(obs[0])

        # Get action name
        action_name = impl._action_names[action] if action < len(impl._action_names) else f"action_{action}"

        # Step environment
        obs, rewards, terminals, truncations, infos = env.step([action])
        step_count += 1

        done = terminals[0] or truncations[0]
        reward = rewards[0]

        # Get current state
        if hasattr(impl, '_cached_state') and impl._cached_state:
            state = impl._cached_state

            # Log comprehensive info
            print(f"STEP {step_count}:")
            print(f"  Action: {action_name} (id={action})")
            print(f"  Position: ({state.agent_row}, {state.agent_col})")
            print(f"  Phase: {state.current_phase.name}")
            print(f"  Glyph: {getattr(state, 'current_glyph', 'unknown')}")
            print(f"  Resources: G={state.germanium} Si={state.silicon} C={state.carbon} O={state.oxygen}")
            print(f"  Decoder: {state.decoder}")
            print(f"  Energy: {state.energy}")
            print(f"  Reward: {reward}")
            print(f"  Done: {done}")
            print()

        if done:
            print(f"Episode completed at step {step_count}")
            break

if __name__ == "__main__":
    run_quick_simulation()
