from omegaconf import OmegaConf
import os
import torch

def load_policy_from_file(path: str, device: str):
    assert path.endswith('.pt'), f"Policy file {path} does not have a .pt extension"
    return torch.load(path, map_location=device, weights_only=False)

def load_policy_from_wandb(cfg: OmegaConf):
    artifact = wandb.use_artifact(cfg.experiment + ":latest", type="model")
    model_path = os.path.join(artifact.download(), "model.pt")
    return load_policy_from_file(model_path, cfg.framework.pufferlib.device)

def load_policy_from_dir(path: str, device: str):
    trainer_state = torch.load(os.path.join(path, 'trainer_state.pt'))
    model_path = os.path.join(path, trainer_state["model_name"])
    return load_policy_from_file(model_path, device)

def load_policy_from_uri(uri: str, device: str):
    if uri.startswith("wandb://"):
        return load_policy_from_wandb(uri)
    elif uri.endswith(".pt"):
        return load_policy_from_file(uri, device)
    else:
        return load_policy_from_dir(uri, device)
