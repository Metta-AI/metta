import logging
import os
import tempfile
from pathlib import Path
from typing import Any, Dict, Optional, TypedDict

import boto3
import torch

from metta.rl.system_config import SystemConfig
from metta.rl.training.optimizer import is_schedulefree_optimizer
from metta.tools.utils.auto_config import auto_policy_storage_decision
from mettagrid.policy.mpt_artifact import MptArtifact, load_mpt
from mettagrid.policy.policy import PolicySpec
from mettagrid.util.file import ParsedURI

logger = logging.getLogger(__name__)


class PolicyMetadata(TypedDict):
    run_name: str
    epoch: int
    uri: str


def key_and_version(uri: str) -> tuple[str, int] | None:
    """Extract run name and epoch from a policy URI."""
    parsed = ParsedURI.parse(uri)
    if parsed.scheme == "mock" and parsed.path:
        return (parsed.path, 0)
    if parsed.scheme == "file" and parsed.local_path:
        file_path = Path(parsed.local_path)
    elif parsed.scheme == "s3" and parsed.key:
        file_path = Path(parsed.key)
    else:
        raise ValueError(f"Could not extract key and version from {uri}")

    return _extract_run_and_epoch(file_path)


def _extract_run_and_epoch(path: Path) -> tuple[str, int] | None:
    stem = path.stem
    if ":v" in stem:
        run_name, suffix = stem.rsplit(":v", 1)
        if run_name and suffix.isdigit():
            return (run_name, int(suffix))
    return None


def _get_all_checkpoints(uri: str) -> list[PolicyMetadata]:
    parsed = ParsedURI.parse(uri)
    if parsed.scheme == "file" and parsed.local_path:
        checkpoint_files = [ckpt for ckpt in parsed.local_path.glob("*.mpt") if ckpt.stem]
    elif parsed.scheme == "s3" and parsed.bucket:
        s3_client = boto3.client("s3")
        prefix = parsed.key or ""
        response = s3_client.list_objects_v2(Bucket=parsed.bucket, Prefix=prefix)

        if response["KeyCount"] == 0:
            return []

        checkpoint_files = [Path(obj["Key"]) for obj in response["Contents"] if obj["Key"].endswith(".mpt")]
    else:
        raise ValueError(f"Cannot get checkpoints from uri: {uri}")

    checkpoint_metadata: list[PolicyMetadata] = []
    for path in checkpoint_files:
        run_and_epoch = _extract_run_and_epoch(path)
        if run_and_epoch:
            path_uri = uri.rstrip("/") + "/" + path.name
            metadata: PolicyMetadata = {
                "run_name": run_and_epoch[0],
                "epoch": run_and_epoch[1],
                "uri": path_uri,
            }
            checkpoint_metadata.append(metadata)

    return checkpoint_metadata


def _latest_checkpoint(uri: str) -> PolicyMetadata | None:
    checkpoints = _get_all_checkpoints(uri)
    if checkpoints:
        return max(checkpoints, key=lambda p: p["epoch"])
    return None


class CheckpointManager:
    """Manages run directories and trainer state checkpointing."""

    def __init__(self, run: str, system_cfg: SystemConfig):
        if not run or not run.strip():
            raise ValueError("Run name cannot be empty")
        if any(char in run for char in [" ", "/", "*", "\\", ":", "<", ">", "|", "?", '"']):
            raise ValueError(f"Run name contains invalid characters: {run}")
        if "__" in run:
            raise ValueError(f"Run name cannot contain '__': {run}")

        self.run = run
        self.run_name = run
        self.run_dir = system_cfg.data_dir / self.run
        self.checkpoint_dir = self.run_dir / "checkpoints"

        os.makedirs(system_cfg.data_dir, exist_ok=True)
        os.makedirs(self.run_dir, exist_ok=True)
        os.makedirs(self.checkpoint_dir, exist_ok=True)

        self._remote_prefix: str | None = None
        if not system_cfg.local_only:
            if system_cfg.remote_prefix:
                parsed = ParsedURI.parse(system_cfg.remote_prefix)
                if parsed.scheme != "s3" or not parsed.bucket or not parsed.key:
                    raise ValueError("remote_prefix must be an s3:// URI with bucket and key prefix")
                key_prefix = parsed.key.rstrip("/")
                self._remote_prefix = f"s3://{parsed.bucket}/{key_prefix}" if key_prefix else f"s3://{parsed.bucket}"

            if self._remote_prefix is None:
                self._setup_remote_prefix()

    def _setup_remote_prefix(self) -> None:
        if self._remote_prefix is not None:
            return

        storage_decision = auto_policy_storage_decision(self.run)
        if storage_decision.remote_prefix:
            self._remote_prefix = storage_decision.remote_prefix
            if storage_decision.reason == "env_override":
                logger.info("Using POLICY_REMOTE_PREFIX: %s", storage_decision.remote_prefix)
            else:
                logger.info("Policies will sync to %s", storage_decision.remote_prefix)
        elif storage_decision.reason == "not_connected":
            logger.info("AWS SSO not detected; policies will remain local.")
        elif storage_decision.reason == "aws_not_enabled":
            logger.info("AWS disabled; policies will remain local.")
        elif storage_decision.reason == "no_base_prefix":
            logger.info("Remote prefix unset; policies will remain local.")

    @property
    def remote_prefix(self) -> str | None:
        return self._remote_prefix

    @property
    def remote_checkpoints_enabled(self) -> bool:
        return self._remote_prefix is not None

    @staticmethod
    def normalize_uri(uri: str) -> str:
        """Convert paths to file:// URIs, resolve :latest."""
        parsed = ParsedURI.parse(uri)
        if uri.endswith(":latest"):
            base_uri = uri[:-7]
            latest = _latest_checkpoint(base_uri)
            if not latest:
                raise ValueError(f"No latest checkpoint found for {base_uri}")
            return latest["uri"]
        return parsed.canonical

    @staticmethod
    def get_policy_metadata(uri: str) -> PolicyMetadata:
        normalized_uri = CheckpointManager.normalize_uri(uri)
        metadata = key_and_version(normalized_uri)
        if not metadata:
            raise ValueError(f"Could not extract metadata from uri {uri}")
        run_name, epoch = metadata
        return {"run_name": run_name, "epoch": epoch, "uri": normalized_uri}

    def get_latest_checkpoint(self) -> str | None:
        local_max = _latest_checkpoint(f"file://{self.checkpoint_dir}")
        remote_max = _latest_checkpoint(self._remote_prefix) if self._remote_prefix else None

        if local_max and remote_max:
            if remote_max["epoch"] > local_max["epoch"]:
                return remote_max["uri"]
            return local_max["uri"]

        if local_max:
            return local_max["uri"]
        if remote_max:
            return remote_max["uri"]
        return None

    def load_trainer_state(self) -> Optional[Dict[str, Any]]:
        trainer_file = self.checkpoint_dir / "trainer_state.pt"
        if not trainer_file.exists():
            return None
        state = torch.load(trainer_file, map_location="cpu", weights_only=False)
        result = {
            "optimizer_state": state.get("optimizer", state.get("optimizer_state")),
            "epoch": state.get("epoch", 0),
            "agent_step": state.get("agent_step", 0),
        }
        if "stopwatch_state" in state:
            result["stopwatch_state"] = state["stopwatch_state"]
        if "curriculum_state" in state:
            result["curriculum_state"] = state["curriculum_state"]
        if "loss_states" in state:
            result["loss_states"] = state["loss_states"]
        return result

    def save_trainer_state(
        self,
        optimizer,
        epoch: int,
        agent_step: int,
        stopwatch_state: Optional[Dict[str, Any]] = None,
        curriculum_state: Optional[Dict[str, Any]] = None,
        loss_states: Optional[Dict[str, Any]] = None,
    ):
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        trainer_file = self.checkpoint_dir / "trainer_state.pt"

        is_schedulefree = is_schedulefree_optimizer(optimizer)
        if is_schedulefree:
            optimizer.eval()

        state: dict[str, Any] = {"optimizer": optimizer.state_dict(), "epoch": epoch, "agent_step": agent_step}
        if stopwatch_state:
            state["stopwatch_state"] = stopwatch_state
        if curriculum_state:
            state["curriculum_state"] = curriculum_state
        if loss_states is not None:
            state["loss_states"] = loss_states

        with tempfile.NamedTemporaryFile(
            dir=self.checkpoint_dir,
            prefix=".trainer_state.pt.",
            suffix=".tmp",
            delete=False,
        ) as tmp_file:
            tmp_path = Path(tmp_file.name)

            try:
                torch.save(state, tmp_path)
                tmp_path.replace(trainer_file)
            except Exception:
                if tmp_path.exists():
                    tmp_path.unlink()
                raise

        if is_schedulefree:
            optimizer.train()

    @staticmethod
    def policy_spec_from_uri(
        uri: str,
        *,
        display_name: str | None = None,
        device: str = "cpu",
        strict: bool = True,
    ) -> PolicySpec:
        normalized_uri = CheckpointManager.normalize_uri(uri)
        return PolicySpec(
            class_path="mettagrid.policy.mpt_policy.MptPolicy",
            init_kwargs={
                "checkpoint_uri": normalized_uri,
                "display_name": display_name,
                "device": device,
                "strict": strict,
            },
        )

    @staticmethod
    def load_artifact_from_uri(uri: str) -> MptArtifact:
        normalized_uri = CheckpointManager.normalize_uri(uri)
        return load_mpt(normalized_uri)
