"""Policy checkpoint management component."""

from __future__ import annotations

import io
import logging
import uuid
import zipfile
from typing import TYPE_CHECKING, Optional

import torch
from pydantic import Field

from metta.agent.policy import Policy, PolicyArchitecture
from metta.rl.checkpoint_manager import CheckpointManager
from metta.rl.training import DistributedHelper, TrainerComponent
from mettagrid.base_config import Config
from mettagrid.policy.mpt_artifact import MptArtifact, load_mpt
from mettagrid.policy.policy import PolicySpec
from mettagrid.policy.policy_env_interface import PolicyEnvInterface
from mettagrid.util.file import write_data
from mettagrid.util.url_schemes import policy_spec_from_uri, resolve_uri

if TYPE_CHECKING:
    from metta.app_backend.clients.stats_client import StatsClient

logger = logging.getLogger(__name__)


class CheckpointerConfig(Config):
    epoch_interval: int = Field(default=30, ge=0)


class Checkpointer(TrainerComponent):
    """Manages policy checkpointing with distributed awareness."""

    def __init__(
        self,
        *,
        config: CheckpointerConfig,
        checkpoint_manager: CheckpointManager,
        distributed_helper: DistributedHelper,
        policy_architecture: PolicyArchitecture,
        stats_client: Optional[StatsClient] = None,
        run_name: str = "",
        git_hash: str | None = None,
    ) -> None:
        super().__init__(epoch_interval=max(1, config.epoch_interval))
        self._master_only = True
        self._config = config
        self._checkpoint_manager = checkpoint_manager
        self._distributed = distributed_helper
        self._policy_architecture: PolicyArchitecture = policy_architecture
        self._stats_client = stats_client
        self._run_name = run_name
        self._git_hash = git_hash
        self._latest_policy_uri: Optional[str] = None
        self._latest_policy_version_id: Optional[uuid.UUID] = None

    def register(self, context) -> None:
        super().register(context)
        context.latest_policy_uri_fn = self.get_latest_policy_uri
        context.latest_policy_uri_value = self.get_latest_policy_uri()
        context.latest_policy_version_id_fn = self.get_latest_policy_version_id
        context.latest_policy_version_id_value = None

    def load_or_create_policy(
        self,
        policy_env_info: PolicyEnvInterface,
        *,
        policy_uri: Optional[str] = None,
    ) -> Policy:
        """Load the latest policy checkpoint or create a new policy."""
        candidate_uri = policy_uri or self._checkpoint_manager.get_latest_checkpoint()
        load_device = torch.device(self._distributed.config.device)

        if self._distributed.is_distributed:
            normalized_uri = None
            if self._distributed.is_master() and candidate_uri:
                normalized_uri = resolve_uri(candidate_uri)
            normalized_uri = self._distributed.broadcast_from_master(normalized_uri)

            if normalized_uri:
                artifact: MptArtifact | None = None
                if self._distributed.is_master():
                    artifact = load_mpt(normalized_uri)

                state_dict = self._distributed.broadcast_from_master(
                    {k: v.cpu() for k, v in artifact.state_dict.items()} if artifact else None
                )
                arch = self._distributed.broadcast_from_master(artifact.architecture if artifact else None)
                action_count = self._distributed.broadcast_from_master(
                    len(policy_env_info.actions.actions()) if self._distributed.is_master() else None
                )

                local_action_count = len(policy_env_info.actions.actions())
                if local_action_count != action_count:
                    raise ValueError(f"Action space mismatch: master={action_count}, rank={local_action_count}")

                policy = arch.make_policy(policy_env_info).to(load_device)
                if hasattr(policy, "initialize_to_environment"):
                    policy.initialize_to_environment(policy_env_info, load_device)
                missing, unexpected = policy.load_state_dict(state_dict, strict=True)
                if missing or unexpected:
                    raise RuntimeError(f"Strict loading failed. Missing: {missing}, Unexpected: {unexpected}")

                if self._distributed.is_master():
                    self._latest_policy_uri = normalized_uri
                    logger.info("Loaded policy from %s", normalized_uri)
                return policy

        if candidate_uri:
            artifact = load_mpt(candidate_uri)
            policy = artifact.instantiate(policy_env_info, load_device)
            self._latest_policy_uri = resolve_uri(candidate_uri)
            logger.info("Loaded policy from %s", candidate_uri)
            return policy

        logger.info("Creating new policy for training run")
        return self._policy_architecture.make_policy(policy_env_info)

    def get_latest_policy_uri(self) -> Optional[str]:
        return self._checkpoint_manager.get_latest_checkpoint() or self._latest_policy_uri

    def get_latest_policy_version_id(self) -> Optional[uuid.UUID]:
        return self._latest_policy_version_id

    def _create_submission_zip(self, policy_spec: PolicySpec) -> bytes:
        """Create a submission zip containing policy_spec.json."""
        buffer = io.BytesIO()
        with zipfile.ZipFile(buffer, "w", zipfile.ZIP_DEFLATED) as zipf:
            zipf.writestr("policy_spec.json", policy_spec.model_dump_json())
        return buffer.getvalue()

    def _upload_submission_zip(self, policy_spec: PolicySpec) -> str | None:
        """Upload a submission zip to S3 and return the s3_path."""
        checkpoint_uri = policy_spec.init_kwargs.get("checkpoint_uri")
        if not checkpoint_uri or not checkpoint_uri.startswith("s3://"):
            return None

        submission_path = checkpoint_uri.replace(".mpt", "-submission.zip")
        zip_data = self._create_submission_zip(policy_spec)
        write_data(submission_path, zip_data, content_type="application/zip")
        logger.info("Uploaded submission zip to %s", submission_path)
        return submission_path

    def _register_policy_version(self, policy_uri: str, epoch: int) -> uuid.UUID | None:
        """Register the policy version with Observatory."""
        if not self._stats_client or not self._run_name:
            return None

        policy_spec = policy_spec_from_uri(policy_uri)
        s3_path = self._upload_submission_zip(policy_spec)

        policy_id = self._stats_client.create_policy(
            name=self._run_name,
            attributes={},
            is_system_policy=False,
        )

        agent_step = getattr(self.context, "agent_step", 0)
        policy_version_id = self._stats_client.create_policy_version(
            policy_id=policy_id.id,
            git_hash=self._git_hash,
            policy_spec=policy_spec.model_dump(mode="json"),
            attributes={"epoch": epoch, "agent_step": agent_step},
            s3_path=s3_path,
        )

        logger.info("Registered policy version %s with Observatory", policy_version_id.id)
        return policy_version_id.id

    def on_epoch_end(self, epoch: int) -> None:
        if not self._distributed.should_checkpoint():
            return
        if epoch % self._config.epoch_interval != 0:
            return
        self._save_policy(epoch)

    def on_training_complete(self) -> None:
        if not self._distributed.should_checkpoint():
            return
        self._save_policy(self.context.epoch)

    def _policy_to_save(self) -> Policy:
        policy: Policy = self.context.policy
        if hasattr(policy, "module"):
            return policy.module
        return policy

    def _save_policy(self, epoch: int) -> None:
        policy = self._policy_to_save()
        uri = self._checkpoint_manager.save_policy_checkpoint(
            state_dict=policy.state_dict(),
            architecture=self._policy_architecture,
            epoch=epoch,
        )

        self._latest_policy_uri = uri
        self.context.latest_policy_uri_value = uri

        policy_version_id = self._register_policy_version(uri, epoch)
        if policy_version_id:
            self._latest_policy_version_id = policy_version_id
            self.context.latest_policy_version_id_value = policy_version_id

        try:
            self.context.latest_saved_policy_epoch = epoch
        except AttributeError:
            logger.debug("Component context missing latest_saved_policy_epoch attribute")
