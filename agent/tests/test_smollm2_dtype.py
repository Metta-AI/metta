#!/usr/bin/env python3
"""Quick test to verify SmolLM2 dtype consistency fix."""

import torch
from tensordict import TensorDict

# Mock minimal environment for testing
class MockEnv:
    def __init__(self):
        self.single_action_space = type('ActionSpace', (), {'nvec': [5, 3, 2]})()
        self.max_action_args = [4, 2, 1]

def test_smollm2_dtype_consistency():
    """Test that SmolLM2 handles dtype mismatches correctly."""
    print("Testing SmolLM2 dtype consistency...")
    
    # Import SmolLM2 from the proper location
    from metta.agent.pytorch.smollm2 import SmolLM2
    
    # Create mock environment
    env = MockEnv()
    
    # Initialize model
    model = SmolLM2(env, model_name="HuggingFaceTB/SmolLM2-135M")
    
    # Check initial dtypes (before initialization)
    llm_dtype = next(model.llm.parameters()).dtype
    token_proj_dtype = next(model.token_projector.parameters()).dtype
    actor_dtype = next(model.actor.parameters()).dtype
    value_dtype = next(model.value.parameters()).dtype
    
    print(f"BEFORE initialization:")
    print(f"  LLM dtype: {llm_dtype}")
    print(f"  Token projector dtype: {token_proj_dtype}")  
    print(f"  Actor dtype: {actor_dtype}")
    print(f"  Value dtype: {value_dtype}")
    
    # Initialize to device (CPU for testing)
    device = torch.device("cpu")
    model.initialize_to_environment(["move", "rotate", "noop"], device)
    
    # Check dtypes AFTER initialization - they should all match now
    llm_dtype_after = next(model.llm.parameters()).dtype
    token_proj_dtype_after = next(model.token_projector.parameters()).dtype
    actor_dtype_after = next(model.actor.parameters()).dtype
    value_dtype_after = next(model.value.parameters()).dtype
    
    print(f"AFTER initialization:")
    print(f"  LLM dtype: {llm_dtype_after}")
    print(f"  Token projector dtype: {token_proj_dtype_after}")  
    print(f"  Actor dtype: {actor_dtype_after}")
    print(f"  Value dtype: {value_dtype_after}")
    
    # Verify all components have matching dtypes
    all_dtypes = [llm_dtype_after, token_proj_dtype_after, actor_dtype_after, value_dtype_after]
    if len(set(all_dtypes)) == 1:
        print(f"✅ All components successfully synchronized to {llm_dtype_after}")
    else:
        print(f"❌ Dtype mismatch still exists: {all_dtypes}")
        return False
    
    # Create test input
    batch_size = 2
    seq_len = 10
    observations = torch.randint(0, 255, (batch_size, seq_len, 3), dtype=torch.uint8)
    
    td = TensorDict({
        "env_obs": observations,
        "done": torch.zeros(batch_size, dtype=torch.bool),
        "reward": torch.zeros(batch_size, dtype=torch.float32)
    }, batch_size=[batch_size])
    
    # Test forward pass - focus specifically on dtype conversion
    try:
        with torch.no_grad():
            # Test just the dtype conversion parts
            observations = observations.float() / 255.0
            token_embeddings = model.token_projector(observations)
            
            print(f"Token embeddings dtype: {token_embeddings.dtype}")
            
            # Test the dtype conversion logic
            llm_dtype = next(model.llm.parameters()).dtype
            if token_embeddings.dtype != llm_dtype:
                token_embeddings = token_embeddings.to(dtype=llm_dtype)
                print(f"✅ Successfully converted token embeddings to {llm_dtype}")
            else:
                print(f"✅ Token embeddings already match LLM dtype: {llm_dtype}")
            
            # Test that we can feed this to the LLM without error
            outputs = model.llm(
                inputs_embeds=token_embeddings,
                output_hidden_states=True,
                return_dict=True,
            )
            print("✅ LLM forward pass successful - no dtype mismatch!")
            
        return True
        
    except RuntimeError as e:
        if "expected mat1 and mat2 to have the same dtype" in str(e):
            print(f"❌ Dtype mismatch still exists: {e}")
            return False
        else:
            print(f"❌ Other error: {e}")
            return False

if __name__ == "__main__":
    success = test_smollm2_dtype_consistency()
    exit(0 if success else 1)