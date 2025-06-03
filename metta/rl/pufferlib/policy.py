from types import SimpleNamespace

import torch
from hydra.utils import instantiate
from omegaconf import DictConfig
from pufferlib.pytorch import sample_logits
from torch import nn


def load_policy(path: str, device: str = "cpu", puffer: DictConfig = None):
    weights = torch.load(path, map_location=device, weights_only=True)

    try:
        num_actions, hidden_size = weights["policy.actor.0.weight"].shape
        num_action_args, _ = weights["policy.actor.1.weight"].shape
        _, obs_channels, _, _ = weights["policy.network.0.weight"].shape
    except Exception as e:
        print(f"Failed automatic parse from weights: {e}")
        # TODO -- fix all magic numbers
        num_actions, num_action_args = 9, 10
        _, obs_channels = 128, 34

    # Create environment namespace
    env = SimpleNamespace(
        single_action_space=SimpleNamespace(nvec=[num_actions, num_action_args]),
        single_observation_space=SimpleNamespace(shape=tuple(torch.zeros((obs_channels, 11, 11)).shape)),
    )

    # Instantiate the policy with correct parameters
    if puffer is None:
        policy = instantiate(
            {
                "_target_": "metta.agent.external.pufferlib.Recurrent",
                "hidden_size": hidden_size,
                "cnn_channels": obs_channels,
            },
            env=env,
        )
    else:
        policy = instantiate(puffer, env=env)

    print(f"Loading {len(weights)} weights into policy")
    
    # Debug: Print what weights we have vs what model expects
    model_state = policy.state_dict()
    print(f"Model expects {len(model_state)} weights")
    
    # Check for weight key mismatches
    missing_keys = []
    unexpected_keys = []
    
    for key in model_state.keys():
        if key not in weights:
            missing_keys.append(key)
    
    for key in weights.keys():
        if key not in model_state:
            unexpected_keys.append(key)
    
    if missing_keys:
        print(f"WARNING: Missing keys in checkpoint: {missing_keys}")
    if unexpected_keys:
        print(f"WARNING: Unexpected keys in checkpoint: {unexpected_keys}")
    
    # Load weights with strict=True to ensure everything loads correctly
    try:
        policy.load_state_dict(weights, strict=True)
        print(f"✓ Successfully loaded all {len(weights)} weights with strict=True")
    except Exception as e:
        print(f"✗ Failed to load weights with strict=True: {e}")
        print("Attempting to identify the specific issue...")
        
        # Try to identify which weights are problematic
        for key in weights.keys():
            if key in model_state:
                if weights[key].shape != model_state[key].shape:
                    print(f"Shape mismatch for {key}: checkpoint {weights[key].shape} vs model {model_state[key].shape}")
            else:
                print(f"Key {key} in checkpoint but not in model")
        
        # As fallback, try non-strict loading but warn
        print("Falling back to non-strict loading...")
        missing_keys, unexpected_keys = policy.load_state_dict(weights, strict=False)
        print(f"Non-strict loading: missing={len(missing_keys)}, unexpected={len(unexpected_keys)}")
        if missing_keys:
            print(f"Missing keys: {missing_keys}")
        if unexpected_keys:
            print(f"Unexpected keys: {unexpected_keys}")

    return PufferAgent(policy)


class PufferAgent(nn.Module):
    def __init__(self, policy: nn.Module):
        super().__init__()
        self.policy = policy
        self.hidden_size = policy.hidden_size
        self.lstm = policy.lstm  # Point to the actual LSTM module, not the entire policy

    def forward(self, obs: torch.Tensor, state, action=None):
        """Uses variable names from LSTMWrapper. Translating for Metta:
        critic -> value
        logprob -> logprob_act
        hidden -> logits then, after sample_logits(), log_sftmx_logits
        """
        # Convert metta PolicyState to PufferLib state format
        puffer_state = self._convert_metta_state_to_puffer(state)
        
        hidden, critic = self.policy(obs, puffer_state)  # using variable names from LSTMWrapper
        action, logprob, logits_entropy = sample_logits(hidden, action)
        
        # Update the original metta state
        self._update_metta_state_from_puffer(state, puffer_state)
        
        # explanation of var names in the docstring above
        return action, logprob, logits_entropy, critic, hidden

    def _convert_metta_state_to_puffer(self, metta_state):
        """Convert metta PolicyState to PufferLib expected state format"""
        # Convert to dictionary format that PufferLib expects
        puffer_state = {
            'lstm_h': None,
            'lstm_c': None
        }
        
        if metta_state is not None:
            # Handle LSTM state
            if hasattr(metta_state, 'lstm_h') and metta_state.lstm_h is not None:
                puffer_state['lstm_h'] = metta_state.lstm_h
            if hasattr(metta_state, 'lstm_c') and metta_state.lstm_c is not None:
                puffer_state['lstm_c'] = metta_state.lstm_c
            
        return puffer_state
    
    def _update_metta_state_from_puffer(self, metta_state, puffer_state):
        """Update metta state with values from puffer state"""
        if metta_state is None:
            return
            
        # Update LSTM state if it exists
        if 'lstm_h' in puffer_state:
            metta_state.lstm_h = puffer_state['lstm_h']
        if 'lstm_c' in puffer_state:
            metta_state.lstm_c = puffer_state['lstm_c']

    def activate_actions(self, actions_names, actions_max_params, device):
        # TODO: this could implement a check that policy's action space matches the environment's
        pass
