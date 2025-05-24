import logging
from types import SimpleNamespace

import torch
from hydra.utils import instantiate
from omegaconf import DictConfig
from pufferlib.cleanrl import sample_logits
from torch import nn

logger = logging.getLogger("pufferlib/policy")


def load_policy(path: str, device: str = "cpu", puffer: DictConfig = None):
    logging.info(f"Loading policy from {path}")
    logging.info(f"Device: {device}")

    weights = torch.load(path, map_location=device, weights_only=True)
    logging.info(f"Loaded weights from {path}")
    logging.info(f"Available weight keys: {list(weights.keys())}")

    try:
        # Log the shapes we're trying to extract from
        if "policy.actor.0.weight" in weights:
            actor_shape = weights["policy.actor.0.weight"].shape
            logging.info(f"policy.actor.0.weight shape: {actor_shape}")
            num_actions, hidden_size = actor_shape
        else:
            logging.error("policy.actor.0.weight not found in weights")
            raise KeyError("policy.actor.0.weight not found")

        if "policy.actor.1.weight" in weights:
            actor1_shape = weights["policy.actor.1.weight"].shape
            logging.info(f"policy.actor.1.weight shape: {actor1_shape}")
            num_action_args, _ = actor1_shape
        else:
            logging.error("policy.actor.1.weight not found in weights")
            raise KeyError("policy.actor.1.weight not found")

        if "policy.network.0.weight" in weights:
            network_shape = weights["policy.network.0.weight"].shape
            logging.info(f"policy.network.0.weight shape: {network_shape}")
            _, obs_channels, _, _ = network_shape
        else:
            logging.error("policy.network.0.weight not found in weights")
            # Try the PolicyRecord approach as fallback
            if "components.cnn1._net.0.weight" in weights:
                cnn_shape = weights["components.cnn1._net.0.weight"].shape
                logging.info(f"components.cnn1._net.0.weight shape: {cnn_shape}")
                obs_channels = cnn_shape[1]
                logging.info(f"Using obs_channels={obs_channels} from components.cnn1._net.0.weight")
            else:
                logging.error("components.cnn1._net.0.weight also not found")
                raise KeyError("Neither policy.network.0.weight nor components.cnn1._net.0.weight found")

        logging.info(
            f"Extracted parameters: num_actions={num_actions}, num_action_args={num_action_args}, "
            f"obs_channels={obs_channels}, hidden_size={hidden_size}"
        )

    except Exception as e:
        logging.error(f"Failed automatic parse from weights: {e}")
        logging.error(f"Exception type: {type(e).__name__}")

        # Show some weight shapes for debugging
        for key, weight in list(weights.items())[:10]:  # Show first 10 for brevity
            logging.info(f"Weight {key}: shape {weight.shape}")

        if len(weights) > 10:
            logging.info(f"... and {len(weights) - 10} more weights")

        # TODO -- fix all magic numbers
        num_actions, num_action_args = 9, 10
        obs_channels = 26  # Changed from original which had _, obs_channels = 128, 34

        logging.warning(
            f"Using fallback values: num_actions={num_actions}, num_action_args={num_action_args}, "
            f"obs_channels={obs_channels}"
        )

    # Create environment namespace
    obs_shape = tuple([obs_channels, 11, 11])  # Simplified without torch.tensor conversion
    logging.info(f"Creating environment with observation shape: {obs_shape}")
    logging.info(f"Action space nvec: [{num_actions}, {num_action_args}]")

    env = SimpleNamespace(
        single_action_space=SimpleNamespace(nvec=[num_actions, num_action_args]),
        single_observation_space=SimpleNamespace(shape=obs_shape),
    )

    logging.info("Instantiating policy with puffer config")
    policy = instantiate(puffer, env=env, policy=None)

    logging.info("Loading state dict into policy")
    policy.load_state_dict(weights)

    logging.info(f"Converting to PufferAgent and moving to device {device}")
    policy = PufferAgent(policy).to(device)

    logging.info("Policy loading completed successfully")
    return policy


class PufferAgent(nn.Module):
    def __init__(self, policy: nn.Module):
        super().__init__()
        self.policy = policy
        self.hidden_size = policy.hidden_size
        self.lstm = policy

    def forward(self, obs: torch.Tensor, state, action=None):
        """Uses variable names from LSTMWrapper. Translating for Metta:
        critic -> value
        logprob -> logprob_act
        hidden -> logits then, after sample_logits(), log_sftmx_logits
        """
        hidden, critic = self.policy(obs, state)  # using variable names from LSTMWrapper
        action, logprob, logits_entropy = sample_logits(hidden, action)
        # explanation of var names in the docstring above
        return action, logprob, logits_entropy, critic, hidden

    def activate_actions(self, actions_names, actions_max_params, device):
        # TODO: this could implement a check that policy's action space matches the environment's
        pass
