#!/usr/bin/env python3

import logging
import numpy as np
from packages.cogames.src.cogames.cogs_vs_clips.missions import get_map
from packages.cogames.src.cogames.cogs_vs_clips.eval_missions import ClipOxygen
from packages.mettagrid.python.src.mettagrid import MettaGridEnv
from packages.cogames.src.cogames.policy.scripted_agent import ScriptedAgentPolicy

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def test_correct_observation():
    """Test with the correct observation structure."""
    
    # Set up mission and environment
    map_builder = get_map("eval_clip_oxygen.map")
    mission = ClipOxygen()
    mission_instance = mission.instantiate(map_builder, num_cogs=1)
    env_cfg = mission_instance.make_env()
    env = MettaGridEnv(env_cfg)
    
    # Create agent
    policy = ScriptedAgentPolicy(env=env)
    agent = policy.agent_policy(agent_id=0)
    
    obs, _ = env.reset()
    
    print("=== CORRECT OBSERVATION TEST ===")
    print(f"Observation type: {type(obs)}")
    print(f"Observation length: {len(obs) if hasattr(obs, '__len__') else 'N/A'}")
    
    # Get feature mappings
    clipped_feature_id = policy._impl._feature_name_to_id.get("clipped", 15)
    type_id_feature = policy._impl._feature_name_to_id.get("type_id", 0)
    print(f"Clipped feature ID: {clipped_feature_id}")
    print(f"Type ID feature: {type_id_feature}")
    
    # Track when we get to the unclipping phase
    unclip_started = False
    before_unclip_obs = None
    after_unclip_obs = None
    
    for step in range(50):
        action = agent.step(obs[0])
        obs, rewards, terminals, truncations, infos = env.step([action])
        
        if hasattr(agent, '_state'):
            state = agent._state
            
            # Check if we're in UNCLIP_STATION phase
            if state.current_phase.name == "UNCLIP_STATION" and not unclip_started:
                print(f"\n*** ENTERED UNCLIP_STATION PHASE at step {step+1} ***")
                unclip_started = True
                
                # Capture observation BEFORE unclipping
                before_unclip_obs = obs[0]  # Access the actual observation data
                print(f"BEFORE unclipping - Agent at ({state.agent_row}, {state.agent_col})")
                
                # Find oxygen extractor observations in the raw data
                oxygen_clipped_before = find_oxygen_clipped_observation(before_unclip_obs, state, clipped_feature_id, type_id_feature)
                print(f"BEFORE unclipping - Oxygen extractor clipped observation: {oxygen_clipped_before}")
                
                # Run unclipping steps
                for unclip_step in range(15):
                    action = agent.step(obs[0])
                    obs, rewards, terminals, truncations, infos = env.step([action])
                    
                    if hasattr(agent, '_state'):
                        state = agent._state
                        print(f"Unclip step {unclip_step+1}: Decoder={state.decoder}, Oxygen={state.oxygen}")
                        
                        # Check if we got oxygen (indicates successful unclipping)
                        if state.oxygen > 0 and after_unclip_obs is None:
                            after_unclip_obs = obs[0]  # Access the actual observation data
                            print(f"AFTER unclipping - Agent at ({state.agent_row}, {state.agent_col})")
                            
                            # Find oxygen extractor observations in the raw data
                            oxygen_clipped_after = find_oxygen_clipped_observation(after_unclip_obs, state, clipped_feature_id, type_id_feature)
                            print(f"AFTER unclipping - Oxygen extractor clipped observation: {oxygen_clipped_after}")
                            
                            # Compare before and after
                            print(f"\n=== COMPARISON ===")
                            print(f"BEFORE: {oxygen_clipped_before}")
                            print(f"AFTER:  {oxygen_clipped_after}")
                            
                            if oxygen_clipped_before is not None and oxygen_clipped_after is not None:
                                if oxygen_clipped_before[3] != oxygen_clipped_after[3]:  # Compare clipped value
                                    print(f"✅ CLIPPED STATUS CHANGED: {oxygen_clipped_before[3]} -> {oxygen_clipped_after[3]}")
                                else:
                                    print(f"❌ CLIPPED STATUS UNCHANGED: {oxygen_clipped_before[3]}")
                            else:
                                print(f"❌ Could not find oxygen extractor observations")
                            
                            break
                    
                    if terminals[0] or truncations[0]:
                        print("Episode ended")
                        break
                
                break
            
        if terminals[0] or truncations[0]:
            print("Episode ended")
            break
    
    print("\n=== TEST COMPLETE ===")

def find_oxygen_clipped_observation(obs, state, clipped_feature_id, type_id_feature):
    """Find the oxygen extractor's clipped observation in raw data using correct structure."""
    oxygen_observations = []
    
    # Use the same parsing logic as the agent
    for tok in obs:
        if len(tok) >= 3:
            # Decode local coords (same as agent)
            packed = int(tok[0])
            obs_r = packed >> 4
            obs_c = packed & 0x0F
            
            # Convert to absolute map coords (same as agent)
            map_r = obs_r - 10 + state.agent_row  # OBS_HEIGHT_RADIUS = 10
            map_c = obs_c - 10 + state.agent_col  # OBS_WIDTH_RADIUS = 10
            
            # Check if this is the oxygen extractor position (15, 16)
            if map_r == 15 and map_c == 16:
                feature_id = int(tok[1])
                feature_value = int(tok[2])
                oxygen_observations.append((map_r, map_c, feature_id, feature_value))
    
    # Look for the clipped feature specifically
    for r, c, fid, val in oxygen_observations:
        if fid == clipped_feature_id:
            return (r, c, fid, val)
    
    return None

if __name__ == "__main__":
    test_correct_observation()
