#!/usr/bin/env python3
"""Comprehensive simulation with detailed logging to file."""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'packages', 'cogames', 'src'))

from cogames.cogs_vs_clips.eval_missions import ClipOxygen
from cogames.cogs_vs_clips.missions import get_map
from cogames.policy.scripted_agent import ScriptedAgentPolicy
from mettagrid import MettaGridEnv
import logging

def run_comprehensive_simulation():
    """Run simulation with comprehensive logging to file."""
    
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
    max_steps = 200
    
    # Open log file
    with open('simulation_log.txt', 'w') as f:
        f.write("=== COMPREHENSIVE SIMULATION LOG ===\n")
        f.write(f"Max steps: {max_steps}\n")
        f.write(f"Initial observation shape: {obs[0].shape}\n\n")
        
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
                f.write(f"STEP {step_count}:\n")
                f.write(f"  Action: {action_name} (id={action})\n")
                f.write(f"  Position: ({state.agent_row}, {state.agent_col})\n")
                f.write(f"  Phase: {state.current_phase.name}\n")
                f.write(f"  Glyph: {getattr(state, 'current_glyph', 'unknown')}\n")
                f.write(f"  Resources: G={state.germanium} Si={state.silicon} C={state.carbon} O={state.oxygen}\n")
                f.write(f"  Decoder: {state.decoder}\n")
                f.write(f"  Energy: {state.energy}\n")
                f.write(f"  Reward: {reward}\n")
                f.write(f"  Done: {done}\n")
                
                # Check if we're at assembler
                assembler_pos = impl._station_positions.get("assembler")
                if assembler_pos:
                    dist_to_assembler = abs(state.agent_row - assembler_pos[0]) + abs(state.agent_col - assembler_pos[1])
                    f.write(f"  Distance to assembler: {dist_to_assembler} (assembler at {assembler_pos})\n")
                
                # Check if we're at oxygen extractor
                oxygen_extractors = impl.extractor_memory.get_by_type("oxygen")
                if oxygen_extractors:
                    for ext in oxygen_extractors:
                        dist_to_oxygen = abs(state.agent_row - ext.position[0]) + abs(state.agent_col - ext.position[1])
                        f.write(f"  Distance to oxygen extractor: {dist_to_oxygen} (oxygen at {ext.position}, clipped={ext.is_clipped})\n")
                
                f.write("\n")
            else:
                f.write(f"STEP {step_count}: action={action_name} (id={action}) | reward={reward} | done={done}\n\n")
            
            if done:
                f.write(f"Episode completed at step {step_count}\n")
                break
    
    print("Simulation completed. Log saved to simulation_log.txt")

if __name__ == "__main__":
    run_comprehensive_simulation()
