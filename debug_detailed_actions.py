#!/usr/bin/env python3
"""Debug script with detailed action and position logging."""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'packages', 'cogames', 'src'))

from cogames.cogs_vs_clips.eval_missions import ClipOxygen
from cogames.cogs_vs_clips.missions import get_map
from cogames.policy.scripted_agent import ScriptedAgentPolicy
from mettagrid import MettaGridEnv
import logging

# Set up detailed logging
logging.basicConfig(level=logging.INFO, format='%(levelname)s:%(name)s:%(message)s')
logger = logging.getLogger(__name__)

def debug_detailed_actions():
    """Debug with detailed action and position logging."""
    logger.info("=== DETAILED ACTION DEBUG ===")
    
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
    max_steps = 100  # Shorter run for debugging
    
    logger.info(f"Starting detailed debug run with max {max_steps} steps")
    logger.info(f"Initial observation shape: {obs[0].shape}")
    
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
            logger.info(f"STEP {step_count}: action={action_name} (id={action}) | pos=({state.agent_row},{state.agent_col}) | phase={state.current_phase.name} | glyph={getattr(state, 'current_glyph', 'unknown')} | resources=G:{state.germanium} Si:{state.silicon} C:{state.carbon} O:{state.oxygen} | decoder:{state.decoder} | energy:{state.energy}")
        else:
            logger.info(f"STEP {step_count}: action={action_name} (id={action}) | reward={reward} | done={done}")
        
        if done:
            logger.info(f"Episode completed at step {step_count}")
            break
    
    logger.info("Detailed debug completed")

if __name__ == "__main__":
    debug_detailed_actions()
