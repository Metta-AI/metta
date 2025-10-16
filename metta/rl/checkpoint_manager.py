import logging
import os
import tempfile
from pathlib import Path
from typing import Any, Dict, Optional, TypedDict
from zipfile import BadZipFile

import boto3
import torch

from metta.agent.mocks import MockAgent
from metta.agent.policy import Policy, PolicyArchitecture
from metta.rl.policy_artifact import (
    PolicyArtifact,
    load_policy_artifact,
    save_policy_artifact_safetensors,
)
from metta.rl.system_config import SystemConfig
from metta.rl.training.optimizer import is_schedulefree_optimizer
from metta.rl.training.training_environment import GameRules
from metta.tools.utils.auto_config import auto_policy_storage_decision
from metta.utils.file import local_copy, write_file
from metta.utils.uri import ParsedURI

logger = logging.getLogger(__name__)


class PolicyMetadata(TypedDict):
    """Type definition for policy metadata returned by get_policy_metadata."""

    run_name: str
    epoch: int
    uri: str


def key_and_version(uri: str) -> tuple[str, int] | None:
    """Extract key (run name) and version (epoch) from a policy URI.

    Examples:
        "file:///tmp/my_run/checkpoints/my_run:v5.mpt" -> ("my_run", 5)
        "s3://bucket/policies/my_run/checkpoints/my_run:v10.mpt" -> ("my_run", 10)
        "mock://test_agent" -> ("test_agent", 0)
    """
    parsed = ParsedURI.parse(uri)
    if parsed.scheme == "mock":
        # For mock URIs, extract the agent name from the path
        return (parsed.path, 0)
    if parsed.scheme == "file" and parsed.local_path:
        file_path = Path(parsed.local_path)
    elif parsed.scheme == "s3" and parsed.key:
        file_path = Path(parsed.key)
    else:
        raise ValueError(f"Could not extract key and version from {uri}")

    return _extract_run_and_epoch(file_path)


def _extract_run_and_epoch(path: Path) -> tuple[str, int] | None:
    """Infer run name and epoch from a checkpoint path.

    Examples:
        "file:///tmp/my_run/checkpoints/my_run:v5.mpt" -> ("my_run", 5)
        "s3://bucket/policies/my_run/checkpoints/my_run:v10.mpt" -> ("my_run", 10)
    """

    stem = path.stem

    if ":v" in stem:
        run_name, suffix = stem.rsplit(":v", 1)
        if run_name and suffix.isdigit():
            return (run_name, int(suffix))


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

        checkpoint_files: list[Path] = [Path(obj["Key"]) for obj in response["Contents"] if obj["Key"].endswith(".mpt")]
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


def _load_checkpoint_file(path: str, is_pt_file: bool = False) -> PolicyArtifact:
    """Load a checkpoint file, raising FileNotFoundError on corruption."""
    try:
        return load_policy_artifact(Path(path), is_pt_file)
    except FileNotFoundError:
        raise
    except (BadZipFile, ValueError, TypeError) as err:
        raise FileNotFoundError(f"Invalid or corrupted checkpoint file: {path}") from err


class CheckpointManager:
    """Checkpoint manager with filename-embedded metadata."""

    def __init__(
        self,
        run: str,
        system_cfg: SystemConfig,
    ):
        # Validate run name
        if not run or not run.strip():
            raise ValueError("Run name cannot be empty")
        if any(char in run for char in [" ", "/", "*", "\\", ":", "<", ">", "|", "?", '"']):
            raise ValueError(f"Run name contains invalid characters: {run}")
        if "__" in run:
            raise ValueError(f"Run name cannot contain '__' as it's used as a delimiter in checkpoint filenames: {run}")

        self.run = run
        self.run_name = run
        self.run_dir = system_cfg.data_dir / self.run
        self.checkpoint_dir = self.run_dir / "checkpoints"

        os.makedirs(system_cfg.data_dir, exist_ok=True)
        os.makedirs(self.run_dir, exist_ok=True)
        os.makedirs(self.checkpoint_dir, exist_ok=True)

        self._remote_prefix = None
        if not system_cfg.local_only:
            if system_cfg.remote_prefix:
                parsed = ParsedURI.parse(system_cfg.remote_prefix)
                if parsed.scheme != "s3" or not parsed.bucket or not parsed.key:
                    raise ValueError("remote_prefix must be an s3:// URI with bucket and key prefix")
                # Remove trailing slash from prefix for deterministic joins
                key_prefix = parsed.key.rstrip("/")
                self._remote_prefix = f"s3://{parsed.bucket}/{key_prefix}" if key_prefix else f"s3://{parsed.bucket}"

            if self._remote_prefix is None:
                self._setup_remote_prefix()

    def _setup_remote_prefix(self) -> None:
        """Determine and set the remote prefix for policy storage if needed."""
        if self._remote_prefix is None:
            storage_decision = auto_policy_storage_decision(self.run)
            if storage_decision.remote_prefix:
                self._remote_prefix = storage_decision.remote_prefix
                if storage_decision.reason == "env_override":
                    logger.info("Using POLICY_REMOTE_PREFIX for policy storage: %s", storage_decision.remote_prefix)
                else:
                    logger.info(
                        "Policies will sync to %s (Softmax AWS profile detected).",
                        storage_decision.remote_prefix,
                    )
            elif storage_decision.reason == "not_connected":
                logger.info(
                    "Softmax AWS SSO not detected; policies will remain local. "
                    "Run 'aws sso login --profile softmax' then 'metta status --components=aws' to enable uploads."
                )
            elif storage_decision.reason == "aws_not_enabled":
                logger.info(
                    "AWS component disabled; policies will remain local. Run 'metta configure aws' to set up S3."
                )
            elif storage_decision.reason == "no_base_prefix":
                logger.info(
                    "Remote policy prefix unset; policies will remain local. Configure POLICY_REMOTE_PREFIX or run "
                    "'metta configure aws'."
                )

    @property
    def remote_checkpoints_enabled(self) -> bool:
        return self._remote_prefix is not None

    @staticmethod
    def load_from_uri(uri: str, game_rules: GameRules, device: torch.device) -> Policy:
        artifact = CheckpointManager.load_artifact_from_uri(uri)
        return artifact.instantiate(game_rules, device)

    @staticmethod
    def load_artifact_from_uri(uri: str) -> PolicyArtifact:
        """Load a policy from a URI (file://, s3://, or mock://).

        Supports :latest selector for automatic resolution to the most recent checkpoint:
            file:///path/to/run/checkpoints/:latest
            s3://bucket/path/run/checkpoints/:latest
        """
        if uri.startswith(("http://", "https://", "ftp://", "gs://")):
            raise ValueError(f"Invalid URI: {uri}")

        uri = CheckpointManager.normalize_uri(uri)
        parsed = ParsedURI.parse(uri)

        if parsed.scheme == "file" and parsed.local_path is not None:
            path = parsed.local_path
            if path.is_dir():
                checkpoint_file = _latest_checkpoint(f"file://{path}")
                if not checkpoint_file:
                    raise FileNotFoundError(f"No checkpoint files in {uri}")
                local_path = ParsedURI.parse(checkpoint_file["uri"]).local_path
                return _load_checkpoint_file(local_path)  # type: ignore
            if not path.exists():
                raise FileNotFoundError(f"Checkpoint file not found: {path}")
            return _load_checkpoint_file(str(path))

        if parsed.scheme == "s3":
            with local_copy(parsed.canonical) as local_path:
                return _load_checkpoint_file(str(local_path), is_pt_file=Path(parsed.canonical).suffix == ".pt")

        if parsed.scheme == "mock":
            return PolicyArtifact(policy=MockAgent())

        raise ValueError(f"Invalid URI: {uri}")

    @staticmethod
    def normalize_uri(uri: str) -> str:
        """Convert paths to file:// URIs, and resolves :latest"""
        parsed = ParsedURI.parse(uri)
        if uri.endswith(":latest"):
            # Remove ":latest" suffix to get the base URI
            base_uri = uri[:-7]  # remove ":latest"
            # Find the latest checkpoint in the base URI
            latest_checkpoint = _latest_checkpoint(base_uri)
            if not latest_checkpoint:
                raise ValueError(f"No latest checkpoint found for {base_uri}")
            return latest_checkpoint["uri"]
        else:
            return parsed.canonical

    @staticmethod
    def get_policy_metadata(uri: str) -> PolicyMetadata:
        """Extract metadata from policy URI."""
        normalized_uri = CheckpointManager.normalize_uri(uri)
        metadata = key_and_version(normalized_uri)
        if not metadata:
            raise ValueError(f"Could not extract metadata from uri {uri}")
        run_name, epoch = metadata
        return {
            "run_name": run_name,
            "epoch": epoch,
            "uri": normalized_uri,
        }

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

    def save_agent(
        self,
        agent: Policy,
        epoch: int,
        *,
        policy_architecture: PolicyArchitecture,
    ) -> str:
        """Save agent checkpoint to disk and upload to remote storage if configured.

        The serialized artifact always includes the policy weights and architecture metadata.

        Returns URI of saved checkpoint (s3:// if remote prefix configured, otherwise file://).
        """
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        filename = f"{self.run_name}:v{epoch}.mpt"
        checkpoint_path = self.checkpoint_dir / filename

        save_policy_artifact_safetensors(
            checkpoint_path,
            policy_architecture=policy_architecture,
            state_dict=agent.state_dict(),
        )

        remote_uri = None
        if self._remote_prefix:
            remote_uri = f"{self._remote_prefix}/{filename}"
            write_file(remote_uri, str(checkpoint_path))

        if remote_uri:
            return remote_uri
        return f"file://{checkpoint_path.resolve()}"

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

        # For ScheduleFree optimizers, ensure we save in eval mode
        is_schedulefree = is_schedulefree_optimizer(optimizer)
        if is_schedulefree:
            optimizer.eval()

        state = {"optimizer": optimizer.state_dict(), "epoch": epoch, "agent_step": agent_step}
        if stopwatch_state:
            state["stopwatch_state"] = stopwatch_state
        if curriculum_state:
            state["curriculum_state"] = curriculum_state
        if loss_states is not None:
            state["loss_states"] = loss_states

        # Atomic save for trainer state to prevent partial saves
        with tempfile.NamedTemporaryFile(
            dir=self.checkpoint_dir,
            prefix=".trainer_state.pt.",
            suffix=".tmp",
            delete=False,
        ) as tmp_file:
            tmp_path = Path(tmp_file.name)

            try:
                torch.save(state, tmp_path)
                # Atomic move: this operation is atomic on most filesystems
                tmp_path.replace(trainer_file)
            except Exception:
                # Clean up temporary file on error
                if tmp_path.exists():
                    tmp_path.unlink()
                raise

        # Restore train mode after saving for ScheduleFree optimizers
        if is_schedulefree:
            optimizer.train()

    def get_latest_checkpoint(self) -> str | None:
        local_max_checkpoint = _latest_checkpoint(f"file://{self.checkpoint_dir}")
        remote_max_checkpoint = None
        if self._remote_prefix:
            _latest_checkpoint(self._remote_prefix)

        if local_max_checkpoint:
            if remote_max_checkpoint and remote_max_checkpoint["epoch"] > local_max_checkpoint["epoch"]:
                raise ValueError("Invalid setup - trying to resume with a remote checkpoint ahead of local")
            return local_max_checkpoint["uri"]
        elif remote_max_checkpoint:
            return remote_max_checkpoint["uri"]
