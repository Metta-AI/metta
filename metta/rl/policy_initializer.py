import logging
import os
from typing import Any, cast

import torch
from omegaconf import DictConfig

from metta.agent.metta_agent import MettaAgent, PolicyAgent
from metta.agent.policy_record import PolicyRecord
from metta.agent.policy_store import PolicyStore
from metta.agent.util.distribution_utils import get_from_master
from metta.core.distributed_config import DistributedConfig
from metta.mettagrid import MettaGridEnv
from metta.rl.policy_management import (
    validate_policy_environment_match,
    wrap_agent_distributed,
)
from metta.rl.system_config import SystemConfig
from metta.rl.trainer_config import TrainerConfig

logger = logging.getLogger(__name__)


class PolicyInitializer:
    def __init__(
        self,
        agent_cfg: DictConfig,
        system_cfg: SystemConfig,
        trainer_cfg: TrainerConfig,
        metta_grid_env: MettaGridEnv,
        distributed_config: DistributedConfig,
        device: torch.device,
    ):
        self.agent_cfg = agent_cfg
        self.system_cfg = system_cfg
        self.trainer_cfg = trainer_cfg
        self.metta_grid_env = metta_grid_env
        self.device = device
        self.distributed_config = distributed_config

    def make_policy(self) -> PolicyAgent:
        """Create a new policy for the given environment and configuration."""

        return MettaAgent(env=self.metta_grid_env, system_cfg=self.system_cfg, agent_cfg=self.agent_cfg)

    def create_policy_record(
        self,
        policy_store: PolicyStore,
        path: str,
        policy: PolicyAgent,
        checkpoint_file_type: str,
        metadata: dict[str, Any] | None = None,
    ) -> PolicyRecord:
        name = os.path.basename(path)
        metadata = metadata or {}

        # ??  encapsulation
        new_policy_record = PolicyRecord(policy_store._policy_loader, name, f"file://{path}", metadata, policy)
        logger.info(f"Created new policy record to {new_policy_record.uri}")
        return new_policy_record

    def _get_blank_policy(self, policy_store: PolicyStore, policy_path: str) -> PolicyRecord:
        policy = self.make_policy()
        policy_record = self.create_policy_record(
            policy_store,
            path=policy_path,
            policy=policy,
            checkpoint_file_type=self.trainer_cfg.checkpoint.checkpoint_file_type,
        )

        # Only master saves the new policy to disk
        if self.distributed_config.is_master:
            policy_record = policy_store.save(policy_record, self.trainer_cfg.checkpoint.checkpoint_file_type)
            logger.info(f"Master saved new policy to {policy_record.uri}")
        else:
            logger.info("Non-master rank: Created policy structure for DDP sync")

        return policy_record

    def get_blank_policy(self, policy_store: PolicyStore, policy_path: str) -> PolicyRecord:
        policy_record = self._get_blank_policy(policy_store, policy_path)
        pr, _ = self._initialize_policy_for_training(policy_record)
        return pr

    def initialize_policy_for_environment(
        self,
        policy_record: PolicyRecord,
        metta_grid_env: MettaGridEnv,
        device: torch.device,
        restore_feature_mapping: bool = True,
    ) -> None:
        policy = policy_record.policy

        # Restore original_feature_mapping from metadata if available
        if restore_feature_mapping and hasattr(policy, "restore_original_feature_mapping"):
            if "original_feature_mapping" in policy_record.metadata:
                policy.restore_original_feature_mapping(policy_record.metadata["original_feature_mapping"])
                logger.info("Restored original_feature_mapping")

        # Initialize policy to environment
        features = metta_grid_env.get_observation_features()
        policy.initialize_to_environment(features, metta_grid_env.action_names, metta_grid_env.max_action_args, device)

    def _initialize_policy_for_training(self, policy_record: PolicyRecord) -> tuple[PolicyRecord, PolicyAgent]:
        """Initialize a policy record for training with distributed sync, compilation, and DDP wrapping."""
        # Synchronize policy metadata from master using NCCL broadcast of objects.
        # This avoids file I/O on non-master ranks while ensuring consistent metadata.
        if torch.distributed.is_initialized():
            try:
                if policy_record is None:
                    raise RuntimeError("PolicyRecord was not initialized")
                synced_metadata = get_from_master(policy_record.metadata if self.distributed_config.is_master else None)
                if synced_metadata is not None:
                    policy_record.metadata = synced_metadata
            except Exception as e:
                logger.warning(f"Rank {self.distributed_config.rank}: Failed to sync policy metadata from master: {e}")

        if policy_record is None:
            raise RuntimeError("Failed to initialize policy record")

        validate_policy_environment_match(policy_record.policy, self.metta_grid_env)

        # Barrier before compile step
        if torch.distributed.is_initialized():
            torch.distributed.barrier()

        policy: PolicyAgent = policy_record.policy

        # Optional compile step
        # ?? should this be applied in get_blank_policy?
        # here policy_record.policy can diverge from policy - this is confusing

        if self.trainer_cfg.compile:
            logger.info("Compiling policy")
            policy = cast(PolicyAgent, torch.compile(policy, mode=self.trainer_cfg.compile_mode))

        # Wrap in DDP if distributed
        # ?? should this be applied in get_blank_policy?
        if torch.distributed.is_initialized():
            logger.info(f"Initializing DistributedDataParallel on device {self.device}")
            torch.distributed.barrier()
            policy = wrap_agent_distributed(policy, self.device)
            torch.distributed.barrier()

        # Initialize policy for environment after wrapping
        # ?? should this be applied in get_blank_policy?
        self.initialize_policy_for_environment(
            policy_record=policy_record,
            metta_grid_env=self.metta_grid_env,
            device=self.device,
            restore_feature_mapping=True,
        )

        # ?? why does the original code not do this?
        # policy_record.cached_policy = policy
        return policy_record, policy

    def get_initial_policy(
        self, policy_store: PolicyStore, policy_path: str, create_new_policy: bool
    ) -> tuple[PolicyRecord, PolicyAgent]:
        # Now all ranks have the same policy_path and can load/create consistently
        policy_record: PolicyRecord | None = None
        if not create_new_policy:
            logger.info(f"Rank {self.distributed_config.rank}: Loading policy from {policy_path}")
            policy_record = policy_store.policy_record(policy_path)
        else:
            logger.info(f"Rank {self.distributed_config.rank}: No existing policy found, creating new one")
            policy_record = self._get_blank_policy(policy_store, policy_path)

        return self._initialize_policy_for_training(policy_record)
