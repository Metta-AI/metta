"""Agent configurations defined in Python.

These replace the YAML agent configs with Python functions that
return agent component dictionaries.
"""

from typing import Any, Dict


def simple_cnn_agent() -> Dict[str, Any]:
    """Simple CNN-based agent configuration."""
    return {
        "observations": {"obs_key": "grid_obs"},
        "clip_range": 0,
        "analyze_weights_interval": 300,
        "components": {
            # Input processing
            "_obs_": {
                "_target_": "metta.agent.lib.obs_token_to_box_shaper.ObsTokenToBoxShaper",
                "sources": None,
            },
            "obs_normalizer": {
                "_target_": "metta.agent.lib.observation_normalizer.ObservationNormalizer",
                "sources": [{"name": "_obs_"}],
            },
            # CNN layers
            "cnn1": {
                "_target_": "metta.agent.lib.nn_layer_library.Conv2d",
                "sources": [{"name": "obs_normalizer"}],
                "nn_params": {
                    "out_channels": 64,
                    "kernel_size": 5,
                    "stride": 3,
                },
            },
            "cnn2": {
                "_target_": "metta.agent.lib.nn_layer_library.Conv2d",
                "sources": [{"name": "cnn1"}],
                "nn_params": {
                    "out_channels": 64,
                    "kernel_size": 3,
                    "stride": 1,
                },
            },
            # Flatten and FC layers
            "obs_flattener": {
                "_target_": "metta.agent.lib.nn_layer_library.Flatten",
                "sources": [{"name": "cnn2"}],
            },
            "fc1": {
                "_target_": "metta.agent.lib.nn_layer_library.Linear",
                "sources": [{"name": "obs_flattener"}],
                "nn_params": {"out_features": 128},
            },
            "encoded_obs": {
                "_target_": "metta.agent.lib.nn_layer_library.Linear",
                "sources": [{"name": "fc1"}],
                "nn_params": {"out_features": 128},
            },
            # LSTM core
            "_core_": {
                "_target_": "metta.agent.lib.lstm.LSTM",
                "sources": [{"name": "encoded_obs"}],
                "output_size": 128,
                "nn_params": {"num_layers": 2},
            },
            "core_relu": {
                "_target_": "metta.agent.lib.nn_layer_library.ReLU",
                "sources": [{"name": "_core_"}],
            },
            # Critic head
            "critic_1": {
                "_target_": "metta.agent.lib.nn_layer_library.Linear",
                "sources": [{"name": "core_relu"}],
                "nn_params": {"out_features": 1024},
                "nonlinearity": "nn.Tanh",
            },
            "_value_": {
                "_target_": "metta.agent.lib.nn_layer_library.Linear",
                "sources": [{"name": "critic_1"}],
                "nn_params": {"out_features": 1},
                "nonlinearity": None,
            },
            # Actor head
            "actor_1": {
                "_target_": "metta.agent.lib.nn_layer_library.Linear",
                "sources": [{"name": "core_relu"}],
                "nn_params": {"out_features": 512},
            },
            "_action_embeds_": {
                "_target_": "metta.agent.lib.action.ActionEmbedding",
                "sources": None,
                "nn_params": {
                    "num_embeddings": 100,
                    "embedding_dim": 16,
                },
            },
            "_action_": {
                "_target_": "metta.agent.lib.actor.MettaActorSingleHead",
                "sources": [
                    {"name": "actor_1"},
                    {"name": "_action_embeds_"},
                ],
            },
        },
    }


def large_cnn_agent() -> Dict[str, Any]:
    """Larger CNN-based agent with more capacity."""
    return {
        "observations": {"obs_key": "grid_obs"},
        "clip_range": 0,
        "analyze_weights_interval": 300,
        "components": {
            # Input processing
            "_obs_": {
                "_target_": "metta.agent.lib.obs_token_to_box_shaper.ObsTokenToBoxShaper",
                "sources": None,
            },
            "obs_normalizer": {
                "_target_": "metta.agent.lib.observation_normalizer.ObservationNormalizer",
                "sources": [{"name": "_obs_"}],
            },
            # Deeper CNN layers
            "cnn1": {
                "_target_": "metta.agent.lib.nn_layer_library.Conv2d",
                "sources": [{"name": "obs_normalizer"}],
                "nn_params": {
                    "out_channels": 128,
                    "kernel_size": 5,
                    "stride": 2,
                },
            },
            "cnn2": {
                "_target_": "metta.agent.lib.nn_layer_library.Conv2d",
                "sources": [{"name": "cnn1"}],
                "nn_params": {
                    "out_channels": 128,
                    "kernel_size": 3,
                    "stride": 2,
                },
            },
            "cnn3": {
                "_target_": "metta.agent.lib.nn_layer_library.Conv2d",
                "sources": [{"name": "cnn2"}],
                "nn_params": {
                    "out_channels": 128,
                    "kernel_size": 3,
                    "stride": 1,
                },
            },
            # Flatten and FC layers
            "obs_flattener": {
                "_target_": "metta.agent.lib.nn_layer_library.Flatten",
                "sources": [{"name": "cnn3"}],
            },
            "fc1": {
                "_target_": "metta.agent.lib.nn_layer_library.Linear",
                "sources": [{"name": "obs_flattener"}],
                "nn_params": {"out_features": 512},
            },
            # LSTM core
            "_core_": {
                "_target_": "metta.agent.lib.lstm.LSTM",
                "sources": [{"name": "fc1"}],
                "output_size": 512,
                "nn_params": {"num_layers": 2},
            },
            # Critic head
            "critic_1": {
                "_target_": "metta.agent.lib.nn_layer_library.Linear",
                "sources": [{"name": "_core_"}],
                "nn_params": {"out_features": 2048},
                "nonlinearity": "nn.Tanh",
            },
            "_value_": {
                "_target_": "metta.agent.lib.nn_layer_library.Linear",
                "sources": [{"name": "critic_1"}],
                "nn_params": {"out_features": 1},
                "nonlinearity": None,
            },
            # Actor head
            "actor_1": {
                "_target_": "metta.agent.lib.nn_layer_library.Linear",
                "sources": [{"name": "_core_"}],
                "nn_params": {"out_features": 1024},
            },
            "_action_embeds_": {
                "_target_": "metta.agent.lib.action.ActionEmbedding",
                "sources": None,
                "nn_params": {
                    "num_embeddings": 100,
                    "embedding_dim": 32,
                },
            },
            "_action_": {
                "_target_": "metta.agent.lib.actor.MettaActorSingleHead",
                "sources": [
                    {"name": "actor_1"},
                    {"name": "_action_embeds_"},
                ],
            },
        },
    }


def attention_agent() -> Dict[str, Any]:
    """Agent with self-attention layers."""
    return {
        "observations": {"obs_key": "grid_obs"},
        "clip_range": 0,
        "analyze_weights_interval": 300,
        "components": {
            # Input processing
            "_obs_": {
                "_target_": "metta.agent.lib.obs_token_to_box_shaper.ObsTokenToBoxShaper",
                "sources": None,
            },
            "obs_normalizer": {
                "_target_": "metta.agent.lib.observation_normalizer.ObservationNormalizer",
                "sources": [{"name": "_obs_"}],
            },
            # CNN backbone
            "cnn1": {
                "_target_": "metta.agent.lib.nn_layer_library.Conv2d",
                "sources": [{"name": "obs_normalizer"}],
                "nn_params": {
                    "out_channels": 64,
                    "kernel_size": 5,
                    "stride": 2,
                },
            },
            "cnn2": {
                "_target_": "metta.agent.lib.nn_layer_library.Conv2d",
                "sources": [{"name": "cnn1"}],
                "nn_params": {
                    "out_channels": 128,
                    "kernel_size": 3,
                    "stride": 2,
                },
            },
            # Self-attention
            "spatial_features": {
                "_target_": "metta.agent.lib.nn_layer_library.Reshape",
                "sources": [{"name": "cnn2"}],
                "shape": [-1, 128],  # Flatten spatial dims
            },
            "self_attention": {
                "_target_": "metta.agent.lib.attention.SelfAttention",
                "sources": [{"name": "spatial_features"}],
                "nn_params": {
                    "embed_dim": 128,
                    "num_heads": 8,
                },
            },
            "attention_pool": {
                "_target_": "metta.agent.lib.nn_layer_library.GlobalAvgPool1d",
                "sources": [{"name": "self_attention"}],
            },
            # LSTM core
            "_core_": {
                "_target_": "metta.agent.lib.lstm.LSTM",
                "sources": [{"name": "attention_pool"}],
                "output_size": 256,
                "nn_params": {"num_layers": 2},
            },
            # Heads
            "critic_1": {
                "_target_": "metta.agent.lib.nn_layer_library.Linear",
                "sources": [{"name": "_core_"}],
                "nn_params": {"out_features": 1024},
            },
            "_value_": {
                "_target_": "metta.agent.lib.nn_layer_library.Linear",
                "sources": [{"name": "critic_1"}],
                "nn_params": {"out_features": 1},
            },
            "actor_1": {
                "_target_": "metta.agent.lib.nn_layer_library.Linear",
                "sources": [{"name": "_core_"}],
                "nn_params": {"out_features": 512},
            },
            "_action_embeds_": {
                "_target_": "metta.agent.lib.action.ActionEmbedding",
                "sources": None,
                "nn_params": {
                    "num_embeddings": 100,
                    "embedding_dim": 32,
                },
            },
            "_action_": {
                "_target_": "metta.agent.lib.actor.MettaActorSingleHead",
                "sources": [
                    {"name": "actor_1"},
                    {"name": "_action_embeds_"},
                ],
            },
        },
    }
