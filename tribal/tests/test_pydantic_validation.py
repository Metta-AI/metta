#!/usr/bin/env python3

import sys
from pathlib import Path

# Add tribal to path  
sys.path.insert(0, str(Path(__file__).parent / "tribal" / "bindings" / "generated"))

from metta.sim.tribal_genny import TribalEnvConfig
from pydantic import BaseModel

class TestModel(BaseModel):
    env: TribalEnvConfig

def test_pydantic():
    print("Creating tribal config...")
    tribal_config = TribalEnvConfig(label="test")
    
    print("Testing Pydantic validation...")
    try:
        test_model = TestModel(env=tribal_config)
        print(f"SUCCESS: Pydantic validation worked")
    except Exception as e:
        print(f"ERROR: Pydantic validation failed: {e}")
        import traceback
        traceback.print_exc()
    
    print("Testing model_copy...")
    try:
        copied = tribal_config.model_copy(deep=True)
        print(f"SUCCESS: model_copy worked")
    except Exception as e:
        print(f"ERROR: model_copy failed: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    test_pydantic()