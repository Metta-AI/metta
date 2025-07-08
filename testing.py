import wandb
from omegaconf import OmegaConf

from metta.sweep.protein_metta import MettaProtein

# Setup config
config = OmegaConf.create(
    {
        "sweep": {
            "protein": {"max_suggestion_cost": 3600, "num_random_samples": 50},
            "parameters": {
                "metric": "reward",
                "goal": "maximize",
                "trainer": {
                    "optimizer": {
                        "learning_rate": {
                            "distribution": "log_normal",
                            "min": 1e-5,
                            "max": 1e-2,
                            "scale": "auto",
                            "mean": 3e-4,
                        }
                    },
                    "batch_size": {"distribution": "uniform_pow2", "min": 16, "max": 128, "scale": "auto", "mean": 64},
                },
                "model": {
                    "dropout_rate": {
                        "distribution": "logit_normal",
                        "min": 0.1,
                        "max": 0.8,
                        "scale": "auto",
                        "mean": 0.3,
                    }
                },
            },
        }
    }
)

# Initialize with wandb
wandb.init(project="my_project")
optimizer = MettaProtein(config)

# Get suggestions
suggestion, info = optimizer.suggest()
print(f"Try learning_rate: {suggestion['trainer']['optimizer']['learning_rate']}")
print(f"Try batch_size: {suggestion['trainer']['batch_size']}")
print(f"Try dropout_rate: {suggestion['model']['dropout_rate']}")

# Record results
optimizer.record_observation(objective=0.95, cost=120.0)
