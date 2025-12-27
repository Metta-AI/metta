import logging
import os
import tempfile
import zipfile
from pathlib import Path
from typing import Any, Dict, Optional

import torch
from safetensors.torch import save as save_safetensors

from metta.rl.system_config import SystemConfig
from metta.rl.training.optimizer import is_schedulefree_optimizer
from metta.tools.utils.auto_config import auto_policy_storage_decision
from mettagrid.policy.submission import POLICY_SPEC_FILENAME, SubmissionPolicySpec
from mettagrid.util.file import write_file
from mettagrid.util.uri_resolvers.schemes import resolve_uri

logger = logging.getLogger(__name__)

_WEIGHTS_FILENAME = "weights.safetensors"


def prepare_state_dict_for_save(state_dict: dict) -> dict:
    result: dict = {}
    seen_storage: set[int] = set()
    for key, tensor in state_dict.items():
        value = tensor.detach().cpu()
        data_ptr = value.data_ptr()
        if data_ptr in seen_storage:
            value = value.clone()
        else:
            seen_storage.add(data_ptr)
        result[key] = value
    return result


def write_checkpoint_dir(
    *,
    base_dir: Path,
    run_name: str,
    epoch: int,
    architecture_spec: str,
    state_dict: dict,
) -> Path:
    checkpoint_dir = (base_dir / f"{run_name}:v{epoch}").expanduser().resolve()
    write_checkpoint_bundle(checkpoint_dir, architecture_spec=architecture_spec, state_dict=state_dict)
    return checkpoint_dir


def write_checkpoint_bundle(
    checkpoint_dir: Path,
    *,
    architecture_spec: str,
    state_dict: dict,
) -> None:
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    weights_blob = save_safetensors(prepare_state_dict_for_save(state_dict))
    _write_file_atomic(checkpoint_dir / _WEIGHTS_FILENAME, weights_blob)
    spec = SubmissionPolicySpec(
        class_path=_class_path_from_architecture_spec(architecture_spec),
        data_path=_WEIGHTS_FILENAME,
        init_kwargs={"architecture_spec": architecture_spec},
    )
    _write_file_atomic(checkpoint_dir / POLICY_SPEC_FILENAME, spec.model_dump_json().encode("utf-8"))


def _class_path_from_architecture_spec(architecture_spec: str) -> str:
    return architecture_spec.split("(", 1)[0].strip()


def _write_file_atomic(path: Path, data: bytes) -> None:
    tmp_path = None
    try:
        with tempfile.NamedTemporaryFile(
            dir=path.parent,
            prefix=f".{path.name}.",
            suffix=".tmp",
            delete=False,
        ) as tmp:
            tmp_path = Path(tmp.name)
            tmp.write(data)
        tmp_path.replace(path)
        tmp_path = None
    finally:
        if tmp_path and tmp_path.exists():
            tmp_path.unlink()


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

    @property
    def output_uri(self) -> str:
        if self._remote_prefix:
            return self._remote_prefix
        return f"file://{self.checkpoint_dir}"

    def get_latest_checkpoint(self) -> str | None:
        def try_resolve(uri: str) -> tuple[str, int] | None:
            try:
                parsed = resolve_uri(uri)
                info = parsed.checkpoint_info
                if info:
                    return (parsed.canonical, info[1])
            except (ValueError, FileNotFoundError):
                pass
            return None

        local = try_resolve(f"file://{self.checkpoint_dir}")
        remote = try_resolve(self.output_uri) if self._remote_prefix else None
        candidates = [c for c in [local, remote] if c]
        if not candidates:
            return None
        return max(candidates, key=lambda x: x[1])[0]

    def save_policy_checkpoint(self, state_dict: dict, architecture, epoch: int) -> str:
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        checkpoint_dir = write_checkpoint_dir(
            base_dir=self.checkpoint_dir,
            run_name=self.run_name,
            epoch=epoch,
            architecture_spec=architecture.to_spec(),
            state_dict=state_dict,
        )

        if self._remote_prefix:
            remote_zip = f"{self.output_uri.rstrip('/')}/{checkpoint_dir.name}.zip"
            zip_path = self._create_checkpoint_zip(checkpoint_dir)
            try:
                write_file(remote_zip, str(zip_path), content_type="application/zip")
            finally:
                zip_path.unlink(missing_ok=True)
            logger.debug("Policy checkpoint saved remotely to %s", remote_zip)
            return remote_zip

        logger.debug("Policy checkpoint saved locally to %s", checkpoint_dir.as_uri())
        return checkpoint_dir.as_uri()

    @staticmethod
    def _create_checkpoint_zip(checkpoint_dir: Path) -> Path:
        with tempfile.NamedTemporaryFile(
            dir=checkpoint_dir.parent,
            prefix=f".{checkpoint_dir.name}.",
            suffix=".zip",
            delete=False,
        ) as tmp_file:
            zip_path = Path(tmp_file.name)

        try:
            with zipfile.ZipFile(zip_path, mode="w", compression=zipfile.ZIP_DEFLATED) as zipf:
                for file_path in checkpoint_dir.rglob("*"):
                    if file_path.is_file():
                        zipf.write(file_path, arcname=file_path.relative_to(checkpoint_dir))
        except Exception:
            zip_path.unlink(missing_ok=True)
            raise

        return zip_path

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
        if "avg_reward" in state:
            result["avg_reward"] = state["avg_reward"]
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
        avg_reward: torch.Tensor | float | None = None,
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
        if avg_reward is not None:
            state["avg_reward"] = torch.as_tensor(avg_reward).detach().to(device="cpu")
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
