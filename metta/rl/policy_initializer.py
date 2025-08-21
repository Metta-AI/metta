import logging
import os
from typing import Any

import torch
from omegaconf import DictConfig

from metta.agent.metta_agent import MettaAgent, PolicyAgent
from metta.agent.policy_record import PolicyRecord
from metta.agent.policy_store import PolicyStore
from metta.mettagrid import MettaGridEnv
from metta.rl.system_config import SystemConfig
from metta.rl.trainer_config import CheckpointFileType

logger = logging.getLogger(__name__)


class PolicyInitializer:
    def __init__(
        self,
        agent_cfg: DictConfig,
        system_cfg: SystemConfig,
        metta_grid_env: MettaGridEnv,
        is_master: bool,
        device: torch.device,
    ):
        self.agent_cfg = agent_cfg
        self.system_cfg = system_cfg
        self.metta_grid_env = metta_grid_env
        self.device = device
        self.is_master = is_master

    def make_policy(self) -> PolicyAgent:
        """Create a new policy for the given environment and configuration."""

        return MettaAgent(
            env=self.metta_grid_env,
            system_cfg=self.system_cfg,
            policy_architecture_cfg=self.agent_cfg,  # type: ignore
        )

    def create_nondistributed_policy_record(
        self,
        policy_store: PolicyStore,
        path: str,
        policy: PolicyAgent,
        metadata: dict[str, Any] | None = None,
    ) -> PolicyRecord:
        name = os.path.basename(path)
        metadata = metadata or {}

        pr = PolicyRecord(policy_store, name, f"file://{path}")
        pr.metadata = metadata
        pr.policy = policy
        logger.info(f"Created new policy record to {pr.uri}")
        return pr

    def create_policy_handle(self, policy_store: PolicyStore, policy_path: str) -> PolicyRecord:
        policy = self.make_policy()
        policy_record = self.create_nondistributed_policy_record(
            policy_store,
            path=policy_path,
            policy=policy,
        )

        # Extract checkpoint file type from policy_path suffix
        # ?? should be in some util function
        if policy_path.endswith(".safetensors"):
            file_type: CheckpointFileType = "safetensors"
        elif policy_path.endswith(".pt") or policy_path.endswith(".pth"):
            file_type = "pt"
        else:
            # Default to pt if no extension or unknown extension
            file_type = "pt"
            logger.warning(f"Unknown file extension in policy_path '{policy_path}', defaulting to 'pt'")

        # Only master saves the new policy to disk
        if self.is_master:
            policy_record = policy_store.save(policy_record, file_type)
            logger.info(f"Master saved new policy to {policy_record.uri}")
        else:
            logger.info("Non-master rank: Created policy structure for DDP sync")

        return policy_record

    def initialize_policy_for_environment(
        self,
        policy_record: PolicyRecord,
        is_master: bool,
        restore_feature_mapping: bool = True,
    ) -> None:
        policy = policy_record.policy

        # Restore original_feature_mapping from metadata if available
        if restore_feature_mapping and hasattr(policy, "restore_original_feature_mapping"):
            if "original_feature_mapping" in policy_record.metadata:
                policy.restore_original_feature_mapping(policy_record.metadata["original_feature_mapping"])
                logger.info("Restored original_feature_mapping")

        # Initialize policy to environment
        features = self.metta_grid_env.get_observation_features()
        policy.initialize_to_environment(
            features, self.metta_grid_env.action_names, self.metta_grid_env.max_action_args, self.device
        )

    def get_blank_policy(self) -> PolicyAgent:
        """Get a blank policy for the agent factory."""
        return self.make_policy()

    def get_initial_policy(self) -> tuple[PolicyRecord, PolicyRecord, PolicyAgent]:
        """Get the initial policy record and policy."""
        # This is a simplified version - in practice, you'd need to integrate with checkpoint_manager
        # For now, create a new policy record
        policy = self.make_policy()

        # Create a temporary policy record (this would normally come from checkpoint_manager)
        # This is a placeholder implementation
        from metta.agent.policy_store import PolicyStore

        policy_store = PolicyStore()  # This would need to be passed in

        policy_record = self.create_nondistributed_policy_record(
            policy_store=policy_store,
            path="/tmp/temp_policy.pt",
            policy=policy,
        )

        return policy_record, policy_record, policy
