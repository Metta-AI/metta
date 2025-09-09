#!/usr/bin/env python3
"""
Debug script to test tribal environment attribute access.
"""

import sys
from pathlib import Path

# Add tribal to path
sys.path.insert(0, str(Path(__file__).parent / "tribal" / "bindings" / "generated"))

from metta.sim.tribal_genny import TribalGridEnv

def test_tribal_attrs():
    print("Creating tribal environment...")
    env = TribalGridEnv()
    
    print("Testing attribute access...")
    try:
        print(f"action_names: {env.action_names}")
    except Exception as e:
        print(f"ERROR accessing action_names: {e}")
    
    try:
        print(f"num_agents: {env.num_agents}")
    except Exception as e:
        print(f"ERROR accessing num_agents: {e}")
    
    try:
        print(f"max_steps: {env.max_steps}")
    except Exception as e:
        print(f"ERROR accessing max_steps: {e}")
    
    try:
        print(f"height: {env.height}")
    except Exception as e:
        print(f"ERROR accessing height: {e}")
    
    try:
        print(f"width: {env.width}")
    except Exception as e:
        print(f"ERROR accessing width: {e}")
    
    try:
        print(f"grid_objects: {env.grid_objects}")
    except Exception as e:
        print(f"ERROR accessing grid_objects: {e}")
        
    try:
        resource_names = getattr(env, 'resource_names', [])
        print(f"resource_names: {resource_names}")
    except Exception as e:
        print(f"ERROR accessing resource_names: {e}")
        
    try:
        object_type_names = getattr(env, 'object_type_names', [])
        print(f"object_type_names: {object_type_names}")
    except Exception as e:
        print(f"ERROR accessing object_type_names: {e}")

    print("Attribute access test completed!")

if __name__ == "__main__":
    test_tribal_attrs()