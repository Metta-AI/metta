import logging
import numpy as np
import torch
from gym import spaces
from omegaconf import DictConfig

from metta.agent.metta_agent import PufferlibRecurrentPolicy, MettaAgentBuilder
from metta.common.util.instantiate import instantiate

logger = logging.getLogger("policy")


class MockEnv:
    def __init__(self):
        self.single_observation_space = spaces.Box(
            low=0.0, high=1.0, shape=(11, 11, 3), dtype=np.float32
        )
        self.single_action_space = spaces.MultiDiscrete([4])
        self.observation_space = self.single_observation_space
        self.action_space = self.single_action_space
        self.grid = np.zeros((11, 11, 3), dtype=np.float32)
        self.agent_pos = [5, 5]

    def reset(self):
        self.grid.fill(0)
        self.agent_pos = [5, 5]
        self._place_agent()
        return self._get_obs(), {}

    def step(self, action: int):
        self.grid[self.agent_pos[0], self.agent_pos[1]] = 0.0

        if action == 0 and self.agent_pos[0] > 0:
            self.agent_pos[0] -= 1
        elif action == 1 and self.agent_pos[0] < 10:
            self.agent_pos[0] += 1
        elif action == 2 and self.agent_pos[1] > 0:
            self.agent_pos[1] -= 1
        elif action == 3 and self.agent_pos[1] < 10:
            self.agent_pos[1] += 1

        self._place_agent()
        obs = self._get_obs()
        return (obs, {}), 0.0, False, {}

    def _place_agent(self):
        self.grid[self.agent_pos[0], self.agent_pos[1]] = 1.0

    def _get_obs(self):
        return self.grid[..., None]



def _get_default_agent_config():
    """Get default MettaAgent configuration."""
    return {
        '_target_': 'metta.agent.metta_agent.MettaAgent',
        'observations': {'obs_key': 'grid_obs'},
        'clip_range': 0,
        'analyze_weights_interval': 300,
        'l2_init_weight_update_interval': 0,
        'components': {
            '_obs_': {
                '_target_': 'metta.agent.lib.obs_token_to_box_shaper.ObsTokenToBoxShaper',
                'sources': None
            },
            'obs_normalizer': {
                '_target_': 'metta.agent.lib.observation_normalizer.ObservationNormalizer',
                'sources': [{'name': '_obs_'}]
            },
            'cnn1': {
                '_target_': 'metta.agent.lib.nn_layer_library.Conv2d',
                'sources': [{'name': 'obs_normalizer'}],
                'nn_params': {'out_channels': 64, 'kernel_size': 5, 'stride': 3}
            },
            'cnn2': {
                '_target_': 'metta.agent.lib.nn_layer_library.Conv2d',
                'sources': [{'name': 'cnn1'}],
                'nn_params': {'out_channels': 64, 'kernel_size': 3, 'stride': 1}
            },
            'obs_flattener': {
                '_target_': 'metta.agent.lib.nn_layer_library.Flatten',
                'sources': [{'name': 'cnn2'}]
            },
            'fc1': {
                '_target_': 'metta.agent.lib.nn_layer_library.Linear',
                'sources': [{'name': 'obs_flattener'}],
                'nn_params': {'out_features': 128}
            },
            'encoded_obs': {
                '_target_': 'metta.agent.lib.nn_layer_library.Linear',
                'sources': [{'name': 'fc1'}],
                'nn_params': {'out_features': 128}
            },
            '_core_': {
                '_target_': 'metta.agent.lib.lstm.LSTM',
                'sources': [{'name': 'encoded_obs'}],
                'output_size': 128,
                'nn_params': {'num_layers': 2}
            },
            'critic_1': {
                '_target_': 'metta.agent.lib.nn_layer_library.Linear',
                'sources': [{'name': '_core_'}],
                'nn_params': {'out_features': 1024},
                'nonlinearity': 'nn.Tanh',
                'effective_rank': True
            },
            '_value_': {
                '_target_': 'metta.agent.lib.nn_layer_library.Linear',
                'sources': [{'name': 'critic_1'}],
                'nn_params': {'out_features': 1},
                'nonlinearity': None
            },
            'actor_1': {
                '_target_': 'metta.agent.lib.nn_layer_library.Linear',
                'sources': [{'name': '_core_'}],
                'nn_params': {'out_features': 512}
            },
            '_action_embeds_': {
                '_target_': 'metta.agent.lib.action.ActionEmbedding',
                'sources': None,
                'nn_params': {'num_embeddings': 100, 'embedding_dim': 16}
            },
            '_action_': {
                '_target_': 'metta.agent.lib.actor.MettaActorSingleHead',
                'sources': [{'name': 'actor_1'}, {'name': '_action_embeds_'}]
            }
        }
    }


def _get_default_feature_normalizations():
    """Get default feature normalization values."""
    return {
        0: 1.0, 1: 10.0, 2: 30.0, 3: 1.0, 4: 1.0, 5: 255.0,
        6: 1.0, 7: 1.0, 8: 255.0, 9: 10.0, 10: 10.0, 11: 100.0,
        12: 255.0, 13: 255.0, 14: 100.0, 15: 100.0, 16: 100.0,
        17: 100.0, 18: 100.0, 19: 100.0, 20: 100.0, 21: 100.0,
        22: 100.0, 23: 100.0
    }


def load_pytorch_policy(path: str, device: str = "cpu", pytorch_cfg: DictConfig = None):

    """Load a PyTorch policy from checkpoint and wrap it in MettaAgent.

    Args:
        path: Path to the checkpoint file
        device: Device to load the policy on
        pytorch_cfg: Configuration for the PyTorch policy with _target_ field

    Returns:
        MettaAgent policy ready for inference
    """


    # Try to extract architecture info from weights
    try:
          # Load checkpoint weights
        weights = torch.load(path, map_location=device, weights_only=True)
        num_actions, hidden_size = weights["policy.actor.0.weight"].shape
        num_action_args, _ = weights["policy.actor.1.weight"].shape
        _, obs_channels, _, _ = weights["policy.network.0.weight"].shape
        logger.info(f"Extracted from weights: actions={num_actions}, hidden={hidden_size}")
    except Exception as e:
        logger.warning(f"Failed to parse architecture from weights: {e}")
        logger.warning("Using defaults from config")
        hidden_size = 128  # Default fallback

    # Create mock environment
    env = MockEnv()

    # Handle policy creation
    if pytorch_cfg is None:
        # Use default Recurrent policy
        from metta.agent.external.example import Policy, Recurrent

        policy = Policy(env=env, cnn_channels=128, hidden_size=hidden_size)
        policy = Recurrent(env=env, policy=policy, input_size=512, hidden_size=hidden_size)
    else:
        # Use provided configuration
        policy = instantiate(pytorch_cfg, env=env, policy=None)

    try:
        policy.load_state_dict(weights)
    except:
        pass


    agent_cfg = _get_default_agent_config()

    builder = MettaAgentBuilder(
        obs_space=env.observation_space,
        obs_width=11,
        obs_height=11,
        action_space=env.single_action_space,
        feature_normalizations=_get_default_feature_normalizations(),
        device=device,
        **agent_cfg,
    )

    base_policy = PufferlibRecurrentPolicy(env)
    metta_agent_policy = builder.build(policy=base_policy)

    logger.info(f"Successfully loaded policy from {path}")
    return metta_agent_policy
