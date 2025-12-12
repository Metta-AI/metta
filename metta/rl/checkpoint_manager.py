import logging
import os
import tempfile
from pathlib import Path
from typing import Any, Dict, Optional

import torch

from metta.rl.system_config import SystemConfig
from metta.rl.training.optimizer import is_schedulefree_optimizer
from metta.tools.utils.auto_config import auto_policy_storage_decision
from metta.rl.mpt_artifact import save_mpt
from mettagrid.policy.policy import PolicySpec
from mettagrid.policy.submission import POLICY_SPEC_FILENAME
from mettagrid.util.file import write_data, write_file
from mettagrid.util.uri_resolvers.schemes import checkpoint_filename, resolve_uri

logger = logging.getLogger(__name__)


class CheckpointManager:
    """Manages run directories and trainer state checkpointing."""

    def __init__(self, run: str, system_cfg: SystemConfig, require_remote_enabled: bool = False):
        if not run or not run.strip():
            raise ValueError("Run name cannot be empty")
        if any(char in run for char in [" ", "/", "*", "\\", ":", "<", ">", "|", "?", '"']):
            raise ValueError(f"Run name contains invalid characters: {run}")
        if "__" in run:
            raise ValueError(f"Run name cannot contain '__': {run}")

        self.run_name = run
        self.run_dir = system_cfg.data_dir / self.run_name
        self.checkpoint_dir = self.run_dir / "checkpoints"

        os.makedirs(system_cfg.data_dir, exist_ok=True)
        os.makedirs(self.run_dir, exist_ok=True)
        os.makedirs(self.checkpoint_dir, exist_ok=True)

        self._remote_prefix: str | None = None
        if not system_cfg.local_only:
            self._setup_remote_prefix()
        if require_remote_enabled and self._remote_prefix is None:
            raise ValueError("Remote checkpoints are required but remote prefix is not set")

    def _setup_remote_prefix(self) -> None:
        storage_decision = auto_policy_storage_decision(self.run_name)
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

    def get_latest_checkpoint(self) -> str | None:
        def _latest_local() -> tuple[str, int] | None:
            if not self.checkpoint_dir.exists():
                return None
            best: tuple[str, int] | None = None
            for candidate in self.checkpoint_dir.iterdir():
                if not candidate.is_dir():
                    continue
                try:
                    parsed = resolve_uri(str(candidate))
                    info = parsed.checkpoint_info
                except (ValueError, FileNotFoundError):
                    continue
                if not info:
                    continue
                if (candidate / POLICY_SPEC_FILENAME).exists():
                    if best is None or info[1] > best[1]:
                        best = (candidate.as_uri(), info[1])
            return best

        def _latest_remote() -> tuple[str, int] | None:
            if not self._remote_prefix:
                return None
            try:
                parsed = resolve_uri(self._remote_prefix)
            except (ValueError, FileNotFoundError):
                return None
            if parsed.scheme != "s3":
                return None

            import boto3

            bucket, key = parsed.canonical[5:].split("/", 1)
            if not key.endswith("/"):
                key = key + "/"
            client = boto3.client("s3")
            response = client.list_objects_v2(Bucket=bucket, Prefix=key)
            best: tuple[str, int] | None = None
            for obj in response.get("Contents", []):
                k = obj["Key"]
                if not k.endswith(POLICY_SPEC_FILENAME):
                    continue
                parts = k.split("/")
                if len(parts) < 2:
                    continue
                run_dir = parts[-2]
                if ":v" not in run_dir:
                    continue
                run, suffix = run_dir.rsplit(":v", 1)
                if run != self.run_name or not suffix.isdigit():
                    continue
                epoch = int(suffix)
                uri = f"s3://{bucket}/{run_dir}"
                if best is None or epoch > best[1]:
                    best = (uri, epoch)
            return best

        candidates = [c for c in [_latest_local(), _latest_remote()] if c]
        if not candidates:
            return None
        return max(candidates, key=lambda x: x[1])[0]

    def _write_policy_spec(self, directory: Path, *, checkpoint_uri: str) -> None:
        spec = PolicySpec(
            class_path="metta.rl.mpt_policy.MptPolicy",
            init_kwargs={"checkpoint_uri": checkpoint_uri, "strict": True},
        )
        spec_path = directory / POLICY_SPEC_FILENAME
        spec_path.write_text(spec.model_dump_json())

    def save_policy_checkpoint(self, state_dict: dict, architecture, epoch: int) -> str:
        dir_name = checkpoint_filename(self.run_name, epoch)
        checkpoint_dir = self.checkpoint_dir / dir_name
        checkpoint_dir.mkdir(parents=True, exist_ok=True)

        mpt_path = checkpoint_dir / "policy.mpt"
        local_mpt_uri = save_mpt(mpt_path, architecture=architecture, state_dict=state_dict)
        self._write_policy_spec(checkpoint_dir, checkpoint_uri=local_mpt_uri)

        local_dir_uri = checkpoint_dir.as_uri()

        if self._remote_prefix:
            remote_dir = f"{self._remote_prefix}/{dir_name}"
            remote_mpt_uri = f"{remote_dir}/policy.mpt"
            write_file(remote_mpt_uri, str(mpt_path))
            write_data(f"{remote_dir}/{POLICY_SPEC_FILENAME}", (checkpoint_dir / POLICY_SPEC_FILENAME).read_bytes())
            logger.debug("Policy checkpoint saved remotely to %s", remote_dir)
            return remote_dir

        logger.debug("Policy checkpoint saved locally to %s", local_dir_uri)
        return local_dir_uri

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
