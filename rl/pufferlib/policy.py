from omegaconf import OmegaConf
import os
import torch
import warnings
import wandb
import random
from copy import deepcopy

def load_policy_from_file(path: str, device: str):
    assert path.endswith('.pt'), f"Policy file {path} does not have a .pt extension"
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", category=FutureWarning)
        policy = torch.load(path, map_location=device, weights_only=False)
        policy.path = path
        if not hasattr(policy, "name"):
            policy.name = "/".join(path.split("/")[-2:])
        if not hasattr(policy, "uri"):
            policy.uri = path
    return policy

def load_policy_from_wandb(uri: str, cfg: OmegaConf, wandb_run):
    artifact_path = uri[len("wandb://"):]
    if "@" in artifact_path:
        if artifact_path.endswith("@"):
            path = artifact_path[:-1]
            selector = cfg.train.policy_selector.range
        else:
            path, selector = artifact_path.split("@")
            selector = int(selector)

        atype, name = path.split("/")
        apath = f"{cfg.wandb.entity}/{cfg.wandb.project}/{name}"
        if not wandb.Api().artifact_collection_exists(type=atype, name=apath):
            return None
        collection = wandb.Api().artifact_collection(type_name=atype, name=apath)
        artifact = select_artifact(collection, cfg, selector)
        if artifact is None:
            return None
        artifact = wandb_run.use_artifact(artifact.qualified_name)
    else:
        artifact = wandb_run.use_artifact(uri[len("wandb://"):])
    data_dir = artifact.download(
        root=os.path.join(cfg.data_dir, "artifacts", artifact.name))
    print(f"Downloaded artifact {artifact.name} to {data_dir}")

    policy = load_policy_from_file(
        os.path.join(data_dir, "model.pt"),
        cfg.device
    )
    policy.uri = f"wandb://{artifact.qualified_name}"
    policy.name = artifact.name
    policy.metadata = deepcopy(artifact.metadata)
    return policy


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
    if policy is None:
        print(f"Failed to load policy from {uri}")
        return None
    print(f"Loaded policy from {uri}")
    if not hasattr(policy, "uri"):
        policy.uri = uri
    if not hasattr(policy, "name"):
        policy.name = extract_policy_name(uri)
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
        wandb_run,
        policy_path,
        name,
        metadata=None, artifact_type="model",
        additional_files=None,
    ):

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
    wandb_run.log_artifact(artifact)
    print(f"Uploaded model to wandb: {artifact.name} to run {wandb_run.id}")
    return artifact

def select_artifact(collection, cfg: OmegaConf, n: int):
    artifacts = list(collection.artifacts())
    selector = cfg.train.policy_selector
    if selector.type == "rand":
        return random.choice(artifacts)
    elif selector.type == "top":
        metric = selector.metric

        top = sorted(artifacts, key=lambda x: x.metadata.get(metric, 0))[-n:]
        if len(top) < n:
            print(f"No artifacts found for {selector}, found {len(top)}")
            return None
        print(f"Top {n} artifacts by {metric}:")
        print(f"{'Artifact':<40} | {metric:<20}")
        print("-" * 62)
        for a in top:
            print(f"{a.name:<40} | {a.metadata.get(metric, 0):<20.4f}")
        return top[-n]
    else:
        raise ValueError(f"Invalid selector {selector}")

def extract_policy_name(uri):
    # Handle URIs starting with 'wandb://'
    if uri.startswith("wandb://"):
        uri = uri[len("wandb://"):]
    if uri.endswith(".pt"):
        return "/".join(uri.split("/")[-2:])
    # Split the URI to extract the policy name
    parts = uri.split('/')
    if len(parts) >= 2:
        model_part = parts[1]
        model_name = model_part.split('@')[0]
        return model_name
    else:
        return uri
