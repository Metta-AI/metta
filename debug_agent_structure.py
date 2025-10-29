#!/usr/bin/env python3

import logging
from packages.cogames.src.cogames.cogs_vs_clips.missions import get_map
from packages.cogames.src.cogames.cogs_vs_clips.eval_missions import ClipOxygen
from packages.mettagrid.python.src.mettagrid import MettaGridEnv
from packages.cogames.src.cogames.policy.scripted_agent import ScriptedAgentPolicy

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def debug_agent_structure():
    """Debug the agent object structure to understand how to access features."""
    
    # Set up mission and environment
    map_builder = get_map("eval_clip_oxygen.map")
    mission = ClipOxygen()
    mission_instance = mission.instantiate(map_builder, num_cogs=1)
    env_cfg = mission_instance.make_env()
    env = MettaGridEnv(env_cfg)
    
    # Create agent
    policy = ScriptedAgentPolicy(env=env)
    agent = policy.agent_policy(agent_id=0)
    
    print("=== AGENT STRUCTURE DEBUG ===")
    print(f"Agent type: {type(agent)}")
    print(f"Agent dir: {[attr for attr in dir(agent) if not attr.startswith('__')]}")
    
    # Check if there's a way to access the policy implementation
    if hasattr(agent, '_policy'):
        print(f"Agent._policy type: {type(agent._policy)}")
        print(f"Agent._policy dir: {[attr for attr in dir(agent._policy) if not attr.startswith('__')]}")
    
    # Check the policy object
    print(f"\nPolicy type: {type(policy)}")
    print(f"Policy dir: {[attr for attr in dir(policy) if not attr.startswith('__')]}")
    
    # Check the _impl attribute
    if hasattr(policy, '_impl'):
        print(f"\nPolicy._impl type: {type(policy._impl)}")
        print(f"Policy._impl dir: {[attr for attr in dir(policy._impl) if not attr.startswith('__')]}")
        
        # Check for feature mapping
        if hasattr(policy._impl, '_feature_name_to_id'):
            print(f"Policy._impl has _feature_name_to_id: {policy._impl._feature_name_to_id}")
        else:
            print("Policy._impl does not have _feature_name_to_id")
    
    # Try to access the feature mapping through the policy
    if hasattr(policy, '_feature_name_to_id'):
        print(f"Policy has _feature_name_to_id: {policy._feature_name_to_id}")
    else:
        print("Policy does not have _feature_name_to_id")

if __name__ == "__main__":
    debug_agent_structure()
