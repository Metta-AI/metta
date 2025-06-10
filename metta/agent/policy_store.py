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

from metta.agent.metta_agent import MettaAgent, make_policy
from metta.rl.pufferlib.policy import load_policy
from metta.util.config import Config
from metta.util.wandb.wandb_context import WandbRun

logger = logging.getLogger("policy_store")


class PolicySelectorConfig(Config):
    type: str = "top"
    metric: str = "score"


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
        elif uri.startswith("puffer://"):
            agents = self._agents_from_puffer(uri[len("puffer://") :])
        else:
            agents = self._agents_from_path(uri)

        if len(agents) == 0:
            raise ValueError(f"No policies found at {uri}")

        logger.info(f"Found {len(agents)} policies at {uri}")

        if selector_type == "all":
            logger.info(f"Returning all {len(agents)} policies")
            return agents
        elif selector_type == "latest":
            selected = [agents[0]]
            logger.info(f"Selected latest policy: {selected[0].name}")
            return selected
        elif selector_type == "rand":
            selected = [random.choice(agents)]
            logger.info(f"Selected random policy: {selected[0].name}")
            return selected
        elif selector_type == "top":
            if (
                "eval_scores" in agents[0].metadata
                and agents[0].metadata["eval_scores"] is not None
                and metric in agents[0].metadata["eval_scores"]
            ):
                # Metric is in eval_scores
                logger.info(f"Found metric '{metric}' in metadata['eval_scores']")
                policy_scores = {p: p.metadata.get("eval_scores", {}).get(metric, None) for p in agents}
            elif metric in agents[0].metadata:
                # Metric is directly in metadata
                logger.info(f"Found metric '{metric}' directly in metadata")
                policy_scores = {p: p.metadata.get(metric, None) for p in agents}
            else:
                # Metric not found anywhere
                logger.warning(
                    f"Metric '{metric}' not found in policy metadata or eval_scores, returning latest policy"
                )
                selected = [agents[0]]
                logger.info(f"Selected latest policy (due to missing metric): {selected[0].name}")
                return selected

            policies_with_scores = [p for p, s in policy_scores.items() if s is not None]

            # If more than 20% of the policies have no score, return the latest policy
            if len(policies_with_scores) < len(agents) * 0.8:
                logger.warning("Too many invalid scores, returning latest policy")
                selected = [agents[0]]  # return latest if metric not found
                logger.info(f"Selected latest policy (due to too many invalid scores): {selected[0].name}")
                return selected

            # Sort by metric score (assuming higher is better)
            def get_policy_score(policy: MettaAgent) -> float:  # Explicitly return a comparable type
                score = policy_scores.get(policy)
                if score is None:
                    return float("-inf")  # Or another appropriate default
                return score

            top = sorted(policies_with_scores, key=get_policy_score)[-n:]

            if len(top) < n:
                logger.warning(f"Only found {len(top)} policies matching criteria, requested {n}")

            logger.info(f"Top {len(top)} policies by {metric}:")
            logger.info(f"{'Policy':<40} | {metric:<20}")
            logger.info("-" * 62)
            for agent in top:
                score = policy_scores[agent]
                logger.info(f"{agent.name:<40} | {score:<20.4f}")

            selected = top[-n:]
            logger.info(f"Selected {len(selected)} top policies by {metric}")
            for i, agent in enumerate(selected):
                logger.info(f"  {i + 1}. {agent.name} (score: {policy_scores[agent]:.4f})")

            return selected
        else:
            raise ValueError(f"Invalid selector type {selector_type}")

    def make_model_name(self, epoch: int):
        return f"model_{epoch:04d}.pt"

    def create(self, env) -> MettaAgent:
        policy = make_policy(env, self._cfg)
        name = self.make_model_name(0)
        path = os.path.join(self._cfg.trainer.checkpoint_dir, name)

        # Update metadata on the created agent
        policy.name = name
        policy.uri = "file://" + path
        policy.local_path = path
        policy.metadata.update(
            {
                "action_names": env.action_names,
                "agent_step": 0,
                "epoch": 0,
                "generation": 0,
                "train_time": 0,
            }
        )

        # Save the initial policy
        self.save(name, path, policy, policy.metadata)

        return policy

    def save(self, name: str, path: str, policy: MettaAgent, metadata: dict):
        logger.info(f"Saving policy to {path}")

        # Update the agent's metadata and info
        policy.name = name
        policy.uri = "file://" + path
        policy.local_path = path
        policy.metadata.update(metadata)

        # Use save_for_training to save complete model for training resumption
        policy.save_for_training(path)

        # Cache the agent
        self._cached_agents[path] = policy
        return policy

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
        else:
            paths.extend([os.path.join(path, p) for p in os.listdir(path) if p.endswith(".pt")])

        return [self._load_from_file(path) for path in paths]

    def _agents_from_wandb_artifact(self, uri: str, version: Optional[str] = None) -> List[MettaAgent]:
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

    def _agents_from_puffer(self, path: str) -> List[MettaAgent]:
        return [self._load_from_puffer(path)]

    def load_from_uri(self, uri: str) -> MettaAgent:
        if uri.startswith("wandb://"):
            return self._load_wandb_artifact(uri[len("wandb://") :])
        if uri.startswith("file://"):
            return self._load_from_file(uri[len("file://") :])
        if uri.startswith("puffer://"):
            return self._load_from_puffer(uri[len("puffer://") :])
        if "://" not in uri:
            return self._load_from_file(uri)

        raise ValueError(f"Invalid URI: {uri}")

    def _make_codebase_backwards_compatible(self):
        """
        torch.load expects the codebase to be in the same structure as when the model was saved.

        We can use this function to alias old layout structures. For now we are just supporting moving
        agent --> metta.agent
        """
        # Memoize
        if getattr(self, "_made_codebase_backwards_compatible", False):
            return
        self._made_codebase_backwards_compatible = True

        # Start with the base module
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

    def _load_from_puffer(self, path: str) -> MettaAgent:
        """Load a puffer policy and wrap it in a MettaAgent."""
        policy = load_policy(path, self._device, puffer=self._cfg.puffer)
        name = os.path.basename(path)

        agent = MettaAgent(
            model=policy,
            model_type="puffer",
            name=name,
            uri="puffer://" + name,
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

        if not path.endswith(".pt") and os.path.isdir(path):
            path = os.path.join(path, os.listdir(path)[-1])
        logger.info(f"Loading policy from {path}")

        assert path.endswith(".pt"), f"Policy file {path} does not have a .pt extension"

        # Try loading with the new training format (complete model)
        try:
            agent = MettaAgent.load_for_training(path, device=self._device)
            if agent.model is not None:
                self._cached_agents[path] = agent
                return agent
            else:
                logger.info("No model in training checkpoint, trying other formats")
        except Exception as e:
            logger.info(f"Failed to load as training format: {e}")

        # Try loading with the new MettaAgent format (for evaluation)
        try:
            agent = MettaAgent.load(path, device=self._device)
            self._cached_agents[path] = agent
            return agent
        except Exception as e:
            logger.info(f"Failed to load as MettaAgent format, trying legacy format: {e}")

        # Try legacy loading for backward compatibility
        self._make_codebase_backwards_compatible()

        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", category=FutureWarning)

            # Load the checkpoint
            loaded_data = torch.load(path, map_location=self._device, weights_only=False)

            # Check if it's an old PolicyRecord object
            if hasattr(loaded_data, "_policy") and hasattr(loaded_data, "metadata"):
                # Convert old PolicyRecord to new MettaAgent
                agent = MettaAgent(
                    model=loaded_data._policy,
                    model_type="brain",  # Assume old models are brain models
                    name=loaded_data.name,
                    uri=loaded_data.uri,
                    metadata=loaded_data.metadata,
                    local_path=path,
                )
                self._cached_agents[path] = agent
                return agent

            # Check if it's a raw model state dict or model object
            elif isinstance(loaded_data, dict) and any(key.startswith("components.") for key in loaded_data.keys()):
                # This looks like a raw state dict from the old MettaAgent/BrainPolicy
                logger.warning(f"Loading raw state dict from {path} - creating new agent wrapper")
                # Create a minimal agent wrapper
                agent = MettaAgent(
                    model=None,  # Can't reconstruct the model from just state dict
                    model_type="brain",
                    name=os.path.basename(path),
                    uri=f"file://{path}",
                    metadata={
                        "epoch": int(os.path.basename(path).split("_")[1].split(".")[0])
                        if "model_" in os.path.basename(path)
                        else 0,
                        "warning": "Loaded from raw state dict - model reconstruction not available",
                    },
                    local_path=path,
                )
                self._cached_agents[path] = agent
                return agent

            # Check if it's a torch model object directly
            elif isinstance(loaded_data, torch.nn.Module):
                logger.warning(f"Loading raw model from {path} - creating new agent wrapper")
                # Try to determine if it's BrainPolicy or generic model
                from metta.agent.brain_policy import BrainPolicy

                if isinstance(loaded_data, BrainPolicy):
                    model = loaded_data
                else:
                    # For generic models, we can't use them directly
                    logger.warning("Generic torch model found - cannot use as MettaAgent model")
                    model = None

                agent = MettaAgent(
                    model=model,
                    model_type="brain",
                    name=os.path.basename(path),
                    uri=f"file://{path}",
                    metadata={"epoch": 0, "warning": "Loaded from raw model - metadata unavailable"},
                    local_path=path,
                )
                self._cached_agents[path] = agent
                return agent
            else:
                raise ValueError(
                    f"Unrecognized file format in {path}. Expected MettaAgent checkpoint, PolicyRecord, state dict, or model object."
                )

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
