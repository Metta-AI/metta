from omegaconf import OmegaConf
import os
import torch
import warnings

def load_policy_from_file(path: str, device: str):
    assert path.endswith('.pt'), f"Policy file {path} does not have a .pt extension"
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", category=FutureWarning)
        policy = torch.load(path, map_location=device, weights_only=False)
        policy.path = path
    return policy

def load_policy_from_wandb(uri: str, cfg: OmegaConf, wandb_run):
    artifact = wandb_run.use_artifact(uri[len("wandb://"):], type="model")
    return load_policy_from_file(artifact.file(
        root=os.path.join(cfg.data_dir, "artifacts")
    ), cfg.device)


def load_policy_from_dir(path: str, device: str):
    trainer_state = torch.load(os.path.join(path, 'trainer_state.pt'))
    model_path = os.path.join(path, trainer_state["model_name"])
    return load_policy_from_file(model_path, device)

def load_policy_from_uri(uri: str, cfg: OmegaConf, wandb_run):
    print(f"Loading policy from {uri}")
    policy = None
    if uri.startswith("wandb://"):
        policy = load_policy_from_wandb(uri, cfg, wandb_run)
    elif uri.endswith(".pt"):
        policy = load_policy_from_file(uri, cfg.device)
    else:
        policy = load_policy_from_dir(uri, cfg.device)
    print(f"Loaded policy from {uri}")
    policy.uri = uri
    return policy

def load_policies_from_dir(path: str, cfg: OmegaConf):
    print(f"Loading policies from {path}")
    policies = []
    for file in os.listdir(path):
        if file.endswith(".pt") and file.startswith("model_"):
            policies.append(load_policy_from_file(os.path.join(path, file), cfg.device))
    print(f"Loaded {len(policies)} policies")
    return policies

def count_params(policy):
    return sum(p.numel() for p in policy.parameters() if p.requires_grad)
