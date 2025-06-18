"""
This file implements a PolicyStore class that manages loading and caching of trained policies.
It provides functionality to:
- Load policies from local files or remote URIs
- Cache loaded policies to avoid reloading
- Select policies based on metadata filters
- Track policy metadata and versioning

The PolicyStore is used by the training system to manage opponent policies and checkpoints.
"""

import collections
import logging
import os
import random
import sys
import warnings
from typing import List, Optional, Union

import torch
import wandb
from omegaconf import DictConfig, ListConfig
from torch import nn

from metta.agent.metta_agent import DistributedMettaAgent, MettaAgent, make_policy
from metta.rl.policy import PytorchAgent, load_policy
from metta.util.config import Config
from metta.util.wandb.wandb_context import WandbRun

logger = logging.getLogger("policy_store")


class PolicySelectorConfig(Config):
    type: str = "top"
    metric: str = "score"


class PolicyRecord:
    def __init__(self, policy_store: "PolicyStore", name: str, uri: str, metadata: dict):
        self._policy_store = policy_store
        self.name = name
        self.uri = uri
        self.metadata = metadata
        self._policy = None
        self._local_path = None

        if self.uri.startswith("file://"):
            self._local_path = self.uri[len("file://") :]

    def policy(self) -> nn.Module:
        if self._policy is None:
            pr = self._policy_store.load_from_uri(self.uri)
            self._policy = pr.policy()
            self._local_path = pr.local_path()
        return self._policy

    def policy_as_metta_agent(self) -> Union[MettaAgent, DistributedMettaAgent, PytorchAgent]:
        """Get the policy as a MettaAgent or DistributedMettaAgent."""
        policy = self.policy()
        if not isinstance(policy, (MettaAgent, DistributedMettaAgent, PytorchAgent)):
            raise TypeError(f"Expected MettaAgent or DistributedMettaAgent, got {type(policy).__name__}")
        return policy

    def num_params(self) -> int:
        return sum(p.numel() for p in self.policy().parameters() if p.requires_grad)

    def local_path(self) -> str | None:
        return self._local_path

    def __repr__(self):
        """Generate a detailed representation of the PolicyRecord with weight shapes."""
        # Basic policy record info
        lines = [f"PolicyRecord(name={self.name}, uri={self.uri})"]

        # Add key metadata if available
        important_keys = ["epoch", "agent_step", "generation", "score"]
        metadata_items = []
        for k in important_keys:
            if k in self.metadata:
                metadata_items.append(f"{k}={self.metadata[k]}")

        if metadata_items:
            lines.append(f"Metadata: {', '.join(metadata_items)}")

        # Load policy if not already loaded
        policy = None
        if self._policy is None:
            try:
                policy = self.policy()
            except Exception as e:
                lines.append(f"Error loading policy: {str(e)}")
                return "\n".join(lines)
        else:
            policy = self._policy

        # Add total parameter count
        total_params = sum(p.numel() for p in policy.parameters())
        trainable_params = sum(p.numel() for p in policy.parameters() if p.requires_grad)
        lines.append(f"Total parameters: {total_params:,} (trainable: {trainable_params:,})")

        # Add module structure with detailed weight shapes
        lines.append("\nModule Structure with Weight Shapes:")

        for name, module in policy.named_modules():
            # Skip top-level module
            if name == "":
                continue

            # Create indentation based on module hierarchy
            indent = "  " * name.count(".")

            # Get module type
            module_type = module.__class__.__name__

            # Start building the module info line
            module_info = f"{indent}{name}: {module_type}"

            # Get parameters for this module (non-recursive)
            params = list(module.named_parameters(recurse=False))

            # Add detailed parameter information
            if params:
                # For common layer types, add specialized shape information
                if isinstance(module, torch.nn.Conv2d):
                    weight = next((p for name, p in params if name == "weight"), None)
                    if weight is not None:
                        out_channels, in_channels, kernel_h, kernel_w = weight.shape
                        module_info += " ["
                        module_info += f"out_channels={out_channels}, "
                        module_info += f"in_channels={in_channels}, "
                        module_info += f"kernel=({kernel_h}, {kernel_w})"
                        module_info += "]"

                elif isinstance(module, torch.nn.Linear):
                    weight = next((p for name, p in params if name == "weight"), None)
                    if weight is not None:
                        out_features, in_features = weight.shape
                        module_info += f" [in_features={in_features}, out_features={out_features}]"

                elif isinstance(module, torch.nn.LSTM):
                    module_info += " ["
                    module_info += f"input_size={module.input_size}, "
                    module_info += f"hidden_size={module.hidden_size}, "
                    module_info += f"num_layers={module.num_layers}"
                    module_info += "]"

                elif isinstance(module, torch.nn.Embedding):
                    weight = next((p for name, p in params if name == "weight"), None)
                    if weight is not None:
                        num_embeddings, embedding_dim = weight.shape
                        module_info += f" [num_embeddings={num_embeddings}, embedding_dim={embedding_dim}]"

                # Add all parameter shapes
                param_shapes = []
                for param_name, param in params:
                    param_shapes.append(f"{param_name}={list(param.shape)}")

                if param_shapes and not any(
                    x in module_info for x in ["out_channels", "in_features", "hidden_size", "num_embeddings"]
                ):
                    module_info += f" ({', '.join(param_shapes)})"

            # Add formatted module info to output
            lines.append(module_info)

        # Add section for buffer shapes (non-parameter tensors like running_mean in BatchNorm)
        buffers = list(policy.named_buffers())
        if buffers:
            lines.append("\nBuffer Shapes:")
            for name, buffer in buffers:
                lines.append(f"  {name}: {list(buffer.shape)}")

        return "\n".join(lines)

    def key_and_version(self) -> tuple[str, int]:
        """
        Extract the policy key and version from the URI.
        TODO: store these on the metadata for new policies,
        eventually read them from the metadata instead of parsing the URI.

        Returns:
            tuple: (policy_key, version)
                - policy_key is the clean name without path or version
                - version is the numeric version or 0 if not present
        """
        # Get the last part after splitting by slash
        base_name = self.uri.split("/")[-1]

        # Check if it has a version number in format ":vNUM"
        if ":" in base_name and ":v" in base_name:
            parts = base_name.split(":v")
            key = parts[0]
            try:
                version = int(parts[1])
            except ValueError:
                version = 0
        else:
            # No version, use the whole thing as key and version = 0
            key = base_name
            version = 0

        return key, version

    def key(self) -> str:
        return self.key_and_version()[0]

    def version(self) -> int:
        return self.key_and_version()[1]


class PolicyStore:
    def __init__(self, cfg: ListConfig | DictConfig, wandb_run: WandbRun | None):
        self._cfg = cfg
        self._device = cfg.device
        self._wandb_run = wandb_run
        self._cached_agents = {}
        self._made_codebase_backwards_compatible = False

    def policy(
        self, policy: Union[str, ListConfig | DictConfig], selector_type: str = "top", n=1, metric="score"
    ) -> MettaAgent:
        if not isinstance(policy, str):
            policy = policy.uri
        agents = self._agents(policy, selector_type, n, metric)
        assert len(agents) == 1, f"Expected 1 policy, got {len(agents)}"
        return agents[0]

    def policies(
        self, policy: Union[str, ListConfig | DictConfig], selector_type: str = "top", n: int = 1, metric: str = "score"
    ) -> List[MettaAgent]:
        if not isinstance(policy, str):
            policy = policy.uri
        return self._agents(policy, selector_type, n=n, metric=metric)

    def _agents(self, uri, selector_type="top", n=1, metric: str = "score"):
        version = None
        if uri.startswith("wandb://"):
            wandb_uri = uri[len("wandb://") :]
            if ":" in wandb_uri:
                wandb_uri, version = wandb_uri.split(":")
            if wandb_uri.startswith("run/"):
                run_id = wandb_uri[len("run/") :]
                agents = self._agents_from_wandb_run(run_id, version)
            elif wandb_uri.startswith("sweep/"):
                sweep_name = wandb_uri[len("sweep/") :]
                agents = self._agents_from_wandb_sweep(sweep_name, version)
            else:
                agents = self._agents_from_wandb_artifact(wandb_uri, version)
        elif uri.startswith("file://"):
            agents = self._agents_from_path(uri[len("file://") :])
        elif uri.startswith("pytorch://"):
            agents = self._agents_from_pytorch(uri[len("pytorch://") :])
        else:
            agents = self._agents_from_path(uri)

        if len(agents) == 0:
            raise ValueError(f"No policies found at {uri}")

        logger.info(f"Found {len(agents)} policies at {uri}")

        selectors = {
            "all": self._select_all,
            "latest": self._select_latest,
            "rand": self._select_rand,
            "top": self._select_top,
        }
        selector = selectors.get(selector_type)

        if selector is None:
            raise ValueError(f"Invalid selector type {selector_type}")

        return selector(agents, n=n, metric=metric)

    def _select_all(self, agents: List[MettaAgent], **kwargs) -> List[MettaAgent]:
        logger.info(f"Returning all {len(agents)} policies")
        return agents

    def _select_latest(self, agents: List[MettaAgent], **kwargs) -> List[MettaAgent]:
        selected = [agents[0]]
        logger.info(f"Selected latest policy: {selected[0].name}")
        return selected

    def _select_rand(self, agents: List[MettaAgent], **kwargs) -> List[MettaAgent]:
        selected = [random.choice(agents)]
        logger.info(f"Selected random policy: {selected[0].name}")
        return selected

    def _select_top(self, agents: List[MettaAgent], n: int = 1, metric: str = "score", **kwargs) -> List[MettaAgent]:
        if not agents:
            logger.warning("No agents to select from")
            return []

        # Check if the metric exists in the first agent's metadata
        first_agent = agents[0]
        if (
            "eval_scores" in first_agent.metadata
            and first_agent.metadata.get("eval_scores") is not None
            and metric in first_agent.metadata["eval_scores"]
        ):
            # Metric is in eval_scores
            logger.info(f"Found metric '{metric}' in metadata['eval_scores']")
            policy_scores = {p: p.metadata.get("eval_scores", {}).get(metric, None) for p in agents}
        elif metric in first_agent.metadata:
            # Metric is directly in metadata
            logger.info(f"Found metric '{metric}' directly in metadata")
            policy_scores = {p: p.metadata.get(metric, None) for p in agents}
        else:
            # Metric not found anywhere
            logger.warning(f"Metric '{metric}' not found in policy metadata or eval_scores, returning latest policy")
            return self._select_latest(agents)

        policies_with_scores = [p for p, s in policy_scores.items() if s is not None]

        # If more than 20% of the policies have no score, return the latest policy
        if len(policies_with_scores) < len(agents) * 0.8:
            logger.warning("Too many invalid scores, returning latest policy")
            return self._select_latest(agents)

        # Sort by metric score (assuming higher is better)
        def get_policy_score(policy: MettaAgent) -> float:  # Explicitly return a comparable type
            score = policy_scores.get(policy)
            if score is None:
                return float("-inf")  # Or another appropriate default
            return score

        top_agents = sorted(policies_with_scores, key=get_policy_score, reverse=True)[:n]

        if len(top_agents) < n:
            logger.warning(f"Only found {len(top_agents)} policies matching criteria, requested {n}")

        logger.info(f"Top {len(top_agents)} policies by {metric}:")
        logger.info(f"{'Policy':<40} | {metric:<20}")
        logger.info("-" * 62)
        for agent in top_agents:
            score = policy_scores[agent]
            logger.info(f"{agent.name:<40} | {score:<20.4f}")

        return top_agents

    def make_model_name(self, epoch: int):
        return f"model_{epoch:04d}.pt"

    def create(self, env) -> MettaAgent:
        agent = make_policy(env, self._cfg)
        name = self.make_model_name(0)
        path = os.path.join(self._cfg.trainer.checkpoint_dir, name)

        # Update metadata on the created agent
        agent.name = name
        agent.uri = "file://" + path
        agent.local_path = path
        agent.metadata.update(
            {
                "action_names": env.action_names,
                "agent_step": 0,
                "epoch": 0,
                "generation": 0,
                "train_time": 0,
            }
        )

        # Save the initial agent
        self.save(name, path, agent, agent.metadata)

        return agent

    def save(self, name: str, path: str, agent: MettaAgent, metadata: dict):
        logger.info(f"Saving policy to {path}")

        # Update the agent's metadata and info
        agent.name = name
        agent.uri = "file://" + path
        agent.local_path = path
        agent.metadata.update(metadata)

        # Use save_for_training to save complete model for training resumption
        agent.save_for_training(path)

        # Cache the agent
        self._cached_agents[path] = agent
        return agent

    def add_to_wandb_run(self, run_id: str, agent: MettaAgent, additional_files=None):
        local_path = agent.local_path
        if local_path is None:
            raise ValueError("MettaAgent has no local path")
        return self.add_to_wandb_artifact(run_id, "model", agent.metadata, local_path, additional_files)

    def add_to_wandb_sweep(self, sweep_name: str, agent: MettaAgent, additional_files=None):
        local_path = agent.local_path
        if local_path is None:
            raise ValueError("MettaAgent has no local path")
        return self.add_to_wandb_artifact(sweep_name, "sweep_model", agent.metadata, local_path, additional_files)

    def add_to_wandb_artifact(self, name: str, type: str, metadata: dict, local_path: str, additional_files=None):
        if self._wandb_run is None:
            raise ValueError("PolicyStore was not initialized with a wandb run")

        additional_files = additional_files or []

        artifact = wandb.Artifact(name, type=type, metadata=metadata)
        artifact.add_file(local_path, name="model.pt")
        for file in additional_files:
            artifact.add_file(file)
        artifact.save()
        artifact.wait()
        logger.info(f"Added artifact {artifact.qualified_name}")
        self._wandb_run.log_artifact(artifact)

    def _agents_from_path(self, path: str) -> List[MettaAgent]:
        paths = []

        if path.endswith(".pt"):
            paths.append(path)
        elif os.path.isdir(path):
            pt_files = sorted([f for f in os.listdir(path) if f.endswith(".pt")])
            if not pt_files:
                logger.warning(f"No .pt files found in directory: {path}")
            paths.extend([os.path.join(path, p) for p in pt_files])
        else:
            # Handle glob patterns
            import glob

            paths = sorted(glob.glob(path))

        return [self._load_from_file(p) for p in paths if p]

    def _agents_from_wandb_artifact(self, uri: str, version: Optional[str] = None) -> List[MettaAgent]:
        # Check if wandb is disabled before proceeding
        if (
            not hasattr(self._cfg, "wandb")
            or not hasattr(self._cfg.wandb, "entity")
            or not hasattr(self._cfg.wandb, "project")
        ):
            raise ValueError(
                f"Cannot load wandb artifact '{uri}' when wandb is disabled (wandb=off). "
                "Either enable wandb or use a local policy URI (file://) instead."
            )
        entity, project, artifact_type, name = uri.split("/")
        path = f"{entity}/{project}/{name}"
        if not wandb.Api().artifact_collection_exists(type=artifact_type, name=path):
            logger.warning(f"No artifact collection found at {uri}")
            return []
        artifact_collection = wandb.Api().artifact_collection(type_name=artifact_type, name=path)

        artifacts = artifact_collection.artifacts()

        if version is not None:
            artifacts = [a for a in artifacts if a.version == version]

        agents = []
        for a in artifacts:
            agent = MettaAgent(name=a.name, uri="wandb://" + a.qualified_name, metadata=a.metadata)
            agents.append(agent)

        return agents

    def _agents_from_wandb_sweep(self, sweep_name: str, version: Optional[str] = None) -> List[MettaAgent]:
        return self._agents_from_wandb_artifact(
            f"{self._cfg.wandb.entity}/{self._cfg.wandb.project}/sweep_model/{sweep_name}", version
        )

    def _agents_from_wandb_run(self, run_id: str, version: Optional[str] = None) -> List[MettaAgent]:
        return self._agents_from_wandb_artifact(
            f"{self._cfg.wandb.entity}/{self._cfg.wandb.project}/model/{run_id}", version
        )

    def _agents_from_pytorch(self, path: str) -> List[MettaAgent]:
        return [self._load_from_pytorch(path)]

    def load_from_uri(self, uri: str) -> MettaAgent:
        if uri.startswith("wandb://"):
            return self._load_wandb_artifact(uri[len("wandb://") :])
        if uri.startswith("file://"):
            return self._load_from_file(uri[len("file://") :])
        if uri.startswith("pytorch://"):
            return self._load_from_pytorch(uri[len("pytorch://") :])
        if "://" not in uri:
            return self._load_from_file(uri)

        raise ValueError(f"Invalid URI: {uri}")

    def _make_codebase_backwards_compatible(self):
        """
        torch.load expects the codebase to be in the same structure as when the model was saved.

        We can use this function to alias old layout structures. For now we are supporting:
        - agent --> metta.agent
        """
        # Memoize
        if getattr(self, "_made_codebase_backwards_compatible", False):
            return
        self._made_codebase_backwards_compatible = True

        # Handle agent --> metta.agent
        sys.modules["agent"] = sys.modules["metta.agent"]
        modules_queue = collections.deque(["metta.agent"])

        processed = set()
        while modules_queue:
            module_name = modules_queue.popleft()
            if module_name in processed:
                continue
            processed.add(module_name)

            if module_name not in sys.modules:
                continue
            module = sys.modules[module_name]
            old_name = module_name.replace("metta.agent", "agent")
            sys.modules[old_name] = module

            # Find all submodules
            for attr_name in dir(module):
                try:
                    attr = getattr(module, attr_name)
                except (ImportError, AttributeError):
                    continue
                if hasattr(attr, "__module__"):
                    attr_module = getattr(attr, "__module__", None)

                    # If it's a module and part of metta.agent, queue it
                    if attr_module and attr_module.startswith("metta.agent"):
                        modules_queue.append(attr_module)

                submodule_name = f"{module_name}.{attr_name}"
                if submodule_name in sys.modules:
                    modules_queue.append(submodule_name)

    def _load_from_pytorch(self, path: str) -> MettaAgent:
        """Load a pytorch policy and wrap it in a MettaAgent."""
        policy = load_policy(path, self._device, puffer=self._cfg.puffer)
        name = os.path.basename(path)

        agent = MettaAgent(
            model=policy,
            model_type="pytorch",
            name=name,
            uri="pytorch://" + name,
            metadata={
                "action_names": [],
                "agent_step": 0,
                "epoch": 0,
                "generation": 0,
                "train_time": 0,
            },
        )

        return agent

    def _load_from_file(self, path: str) -> MettaAgent:
        if path in self._cached_agents:
            return self._cached_agents[path]

        logger.info(f"Loading policy from {path}")
        assert path.endswith(".pt"), f"Policy file {path} does not have a .pt extension"

        # Try loading with modern formats first
        for loader in [MettaAgent.load_for_training, MettaAgent.load]:
            try:
                agent = loader(path, device=self._device)
                if loader == MettaAgent.load_for_training and agent.model is None:
                    logger.info("load_for_training resulted in a null model, trying next loader.")
                    continue
                self._cached_agents[path] = agent
                return agent
            except Exception as e:
                logger.info(f"Failed to load with {loader.__name__}: {e}")

        # Fallback to legacy loading
        logger.info("Modern loaders failed, trying legacy format.")
        self._make_codebase_backwards_compatible()
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", category=FutureWarning)
            loaded_data = torch.load(path, map_location=self._device, weights_only=False)
            agent = self._load_legacy_checkpoint(path, loaded_data)
            if agent:
                self._cached_agents[path] = agent
                return agent

        raise ValueError(f"All methods failed to load policy from {path}")

    def _load_legacy_checkpoint(self, path: str, loaded_data: object) -> Optional[MettaAgent]:
        """Attempt to load a policy from a legacy checkpoint object."""
        agent = None
        if hasattr(loaded_data, "_policy") and hasattr(loaded_data, "metadata"):
            logger.info("Loading from legacy PolicyRecord object.")
            agent = MettaAgent(
                model=loaded_data._policy,
                model_type="brain",
                name=loaded_data.name,
                uri=loaded_data.uri,
                metadata=loaded_data.metadata,
                local_path=path,
            )
        elif isinstance(loaded_data, dict) and any(k.startswith("components.") for k in loaded_data.keys()):
            logger.warning(f"Loading raw state dict from {path}, model cannot be reconstructed.")
            agent = MettaAgent(
                model=None,
                model_type="brain",
                name=os.path.basename(path),
                uri=f"file://{path}",
                metadata={
                    "epoch": int(os.path.basename(path).split("_")[1].split(".")[0])
                    if "model_" in os.path.basename(path)
                    else 0,
                    "warning": "Loaded from raw state dict; model reconstruction not available.",
                },
                local_path=path,
            )
        elif isinstance(loaded_data, torch.nn.Module):
            logger.warning(f"Loading raw model from {path}, some metadata may be missing.")
            from metta.agent.brain_policy import BrainPolicy

            model = loaded_data if isinstance(loaded_data, BrainPolicy) else None
            if model is None:
                logger.warning("Generic torch.nn.Module loaded, cannot be used as MettaAgent model.")

            agent = MettaAgent(
                model=model,
                model_type="brain",
                name=os.path.basename(path),
                uri=f"file://{path}",
                metadata={"epoch": 0, "warning": "Loaded from raw model; metadata unavailable."},
                local_path=path,
            )

        return agent

    def _load_wandb_artifact(self, qualified_name: str):
        logger.info(f"Loading policy from wandb artifact {qualified_name}")

        artifact = wandb.Api().artifact(qualified_name)

        artifact_path = os.path.join(self._cfg.data_dir, "artifacts", artifact.name)

        if not os.path.exists(artifact_path):
            artifact.download(root=artifact_path)

        logger.info(f"Downloaded artifact {artifact.name} to {artifact_path}")

        agent = self._load_from_file(os.path.join(artifact_path, "model.pt"))
        agent.metadata.update(artifact.metadata)
        return agent
