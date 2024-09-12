from omegaconf import OmegaConf
import os
import torch
import wandb
from rl.wandb.wandb import init_wandb

def load_policy_from_file(path: str, device: str):
    assert path.endswith('.pt'), f"Policy file {path} does not have a .pt extension"
    return torch.load(path, map_location=device, weights_only=False)

def load_policy_from_wandb(uri: str, cfg: OmegaConf):
    init_wandb(cfg)

    artifact = wandb.use_artifact(uri[len("wandb://"):], type="model")
    return load_policy_from_file(artifact.file(
        root=os.path.join(cfg.data_dir, "artifacts")
    ), cfg.framework.pufferlib.device)

def load_policy_from_dir(path: str, device: str):
    trainer_state = torch.load(os.path.join(path, 'trainer_state.pt'))
    model_path = os.path.join(path, trainer_state["model_name"])
    return load_policy_from_file(model_path, device)

def load_policy_from_uri(uri: str, cfg: OmegaConf):
    if uri.startswith("wandb://"):
        return load_policy_from_wandb(uri, cfg)
    elif uri.endswith(".pt"):
        return load_policy_from_file(uri, cfg.framework.pufferlib.device)
    else:
        return load_policy_from_dir(uri, cfg.framework.pufferlib.device)
