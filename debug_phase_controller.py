#!/usr/bin/env python3
"""Debug the phase controller with detailed logging."""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'packages', 'cogames', 'src'))

from cogames.cogs_vs_clips.eval_missions import ClipOxygen
from cogames.cogs_vs_clips.missions import get_map
from cogames.policy.scripted_agent import ScriptedAgentPolicy
from mettagrid import MettaGridEnv
import logging

# Set up detailed logging
logging.basicConfig(level=logging.DEBUG, format='%(levelname)s:%(name)s:%(message)s')
logger = logging.getLogger(__name__)

def debug_phase_controller():
    """Debug the phase controller with detailed logging."""
    logger.info("=== DEBUGGING PHASE CONTROLLER ===")

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

    # Run evaluation with detailed logging
    obs = env.reset()
    step_count = 0
    max_steps = 200  # Shorter run for debugging

    logger.info(f"Starting debug run with max {max_steps} steps")
    logger.info(f"Initial observation shape: {obs[0].shape}")

    while step_count < max_steps:
        # Get action
        action = agent.step(obs[0])

        # Step environment
        obs, rewards, terminals, truncations, infos = env.step([action])
        step_count += 1

        done = terminals[0] or truncations[0]
        reward = rewards[0]

        # Debug every 10 steps
        if step_count % 10 == 0 or step_count <= 20:
            logger.info(f"\n=== STEP {step_count} ===")
            logger.info(f"Reward: {reward}, Done: {done}")

            # Access internal state for debugging
            if hasattr(impl, '_cached_state') and impl._cached_state:
                state = impl._cached_state
                logger.info(f"Current phase: {state.current_phase}")
                logger.info(f"Current glyph: {getattr(state, 'current_glyph', 'unknown')}")
                logger.info(f"Energy: {state.energy}")
                logger.info(f"Resources: G={state.germanium} Si={state.silicon} C={state.carbon} O={state.oxygen}")
                logger.info(f"Decoder: {state.decoder}")
                logger.info(f"Agent position: ({state.agent_row}, {state.agent_col})")

                # Check extractor memory
                logger.info(f"Extractor memory:")
                for resource_type, extractors in impl.extractor_memory._extractors.items():
                    logger.info(f"  {resource_type}: {len(extractors)} extractors")
                    for i, ext in enumerate(extractors):
                        logger.info(f"    {i}: pos={ext.position}, clipped={ext.is_clipped}, available={ext.is_available(state.step_count, impl.cooldown_remaining)}")

                # Check station positions
                logger.info(f"Station positions: {impl._station_positions}")

                # Check phase controller state
                logger.info(f"Phase controller current phase: {impl.phase_controller.current()}")
                logger.info(f"Phase controller runtime: {impl.phase_controller._rt}")

        if done:
            logger.info(f"Episode completed at step {step_count}")
            break

    logger.info("Debug completed")

if __name__ == "__main__":
    debug_phase_controller()
