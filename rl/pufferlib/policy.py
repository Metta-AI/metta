from omegaconf import OmegaConf
import os
import torch
import warnings
import wandb

def load_policy_from_file(path: str, device: str):
    assert path.endswith('.pt'), f"Policy file {path} does not have a .pt extension"
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", category=FutureWarning)
        policy = torch.load(path, map_location=device, weights_only=False)
        policy.path = path
    return policy

def load_policy_from_wandb(uri: str, cfg: OmegaConf, wandb_run):
    artifact = wandb_run.use_artifact(uri[len("wandb://"):])
    data_dir = artifact.download(root=os.path.join(cfg.data_dir, "artifacts"))
    return load_policy_from_file(
        os.path.join(data_dir, "model.pt"),
        cfg.device
    )


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

def load_policies_from_wandb(uri: str, cfg: OmegaConf, wandb_run):
    artifact = wandb_run.use_artifact(uri[len("wandb://"):], type="model")
    return load_policies_from_dir(artifact.download(
        root=os.path.join(cfg.data_dir, "artifacts")
    ), cfg)

def count_params(policy):
    return sum(p.numel() for p in policy.parameters() if p.requires_grad)

def upload_policy_to_wandb(
        policy_path,
        name,
        metadata=None, artifact_type="model",
        additional_files=None,
        wandb_run_id=None,
    ):
    wandb_api = wandb.Api()

    artifact = wandb.Artifact(
        name,
        type=artifact_type,
        metadata=metadata or {}
    )
    artifact.add_file(policy_path, name="model.pt")
    if additional_files:
        for file in additional_files:
            artifact.add_file(file)
    artifact.save()
    artifact.wait()
    if wandb_run_id:
        artifact = wandb_api.artifact(artifact.qualified_name)
        wandb_run = wandb_api.run(wandb_run_id)
        wandb_run.log_artifact(artifact)
    print(f"Uploaded model to wandb: {artifact.name} to run {wandb_run_id}")
    return artifact.name
