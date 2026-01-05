import logging
import os
import tempfile
import zipfile
from pathlib import Path
from typing import Any, Dict, Optional

import torch
from safetensors.torch import save_file as save_safetensors_file

from metta.agent.policy import PolicyArchitecture
from metta.rl.system_config import SystemConfig
from metta.rl.training.optimizer import is_schedulefree_optimizer
from metta.tools.utils.auto_config import PolicyStorageDecision, auto_policy_storage_decision
from mettagrid.policy.submission import POLICY_SPEC_FILENAME, SubmissionPolicySpec
from mettagrid.util.file import write_file
from mettagrid.util.uri_resolvers.schemes import resolve_uri

logger = logging.getLogger(__name__)


def write_checkpoint_bundle(
    checkpoint_dir: Path,
    *,
    architecture_spec: str,
    state_dict: dict,
) -> None:
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    weights: dict = {}
    seen_storage: set[int] = set()
    for key, tensor in state_dict.items():
        value = tensor.detach().cpu()
        data_ptr = value.data_ptr()
        if data_ptr in seen_storage:
            value = value.clone()
        else:
            seen_storage.add(data_ptr)
        weights[key] = value
    with tempfile.NamedTemporaryFile(
        dir=checkpoint_dir,
        prefix=".weights.safetensors.",
        suffix=".tmp",
        delete=False,
    ) as tmp:
        tmp_path = Path(tmp.name)
    save_safetensors_file(weights, str(tmp_path))
    tmp_path.replace(checkpoint_dir / "weights.safetensors")
    spec = SubmissionPolicySpec(
        class_path="metta.agent.policy.CheckpointPolicy",
        data_path="weights.safetensors",
        init_kwargs={"architecture_spec": architecture_spec, "device": "cpu"},
    )
    with tempfile.NamedTemporaryFile(
        dir=checkpoint_dir,
        prefix=f".{POLICY_SPEC_FILENAME}.",
        suffix=".tmp",
        delete=False,
    ) as tmp:
        tmp_path = Path(tmp.name)
        tmp.write(spec.model_dump_json().encode("utf-8"))
    tmp_path.replace(checkpoint_dir / POLICY_SPEC_FILENAME)


class CheckpointManager:
    """Manages run directories and trainer state checkpointing."""

    def __init__(
        self,
        run: str,
        system_cfg: SystemConfig,
        require_remote_enabled: bool = False,
        storage_decision: PolicyStorageDecision | None = None,
    ):
        self.run_name = run
        self.run_dir = system_cfg.data_dir / self.run_name
        self.checkpoint_dir = self.run_dir / "checkpoints"

        os.makedirs(system_cfg.data_dir, exist_ok=True)
        os.makedirs(self.run_dir, exist_ok=True)
        os.makedirs(self.checkpoint_dir, exist_ok=True)

        self._remote_prefix: str | None = None
        if not system_cfg.local_only:
            self._setup_remote_prefix(storage_decision)
        if require_remote_enabled and self._remote_prefix is None:
            raise ValueError("Remote checkpoints are required but remote prefix is not set")

    def _setup_remote_prefix(self, storage_decision: PolicyStorageDecision | None = None) -> None:
        if storage_decision is None:
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
        return self._remote_prefix or f"file://{self.checkpoint_dir}"

    def _checkpoint_root(self, slot_id: str | None = None) -> Path:
        if slot_id:
            return self.checkpoint_dir / "slots" / slot_id
        return self.checkpoint_dir

    def _checkpoint_uri_base(self, slot_id: str | None = None) -> str:
        if self._remote_prefix:
            suffix = f"/slots/{slot_id}" if slot_id else ""
            return f"{self._remote_prefix.rstrip('/')}{suffix}"
        return f"file://{self._checkpoint_root(slot_id)}"

    def get_latest_checkpoint(self, slot_id: str | None = None) -> str | None:
        def resolve_candidate(uri: str) -> tuple[str, int] | None:
            parsed = resolve_uri(uri)
            info = parsed.checkpoint_info
            if info:
                return (parsed.canonical, info[1])
            return None

        local = resolve_candidate(f"file://{self._checkpoint_root(slot_id)}")
        remote = resolve_candidate(self._checkpoint_uri_base(slot_id)) if self._remote_prefix else None
        candidates = [c for c in [local, remote] if c]
        return max(candidates, key=lambda x: x[1])[0] if candidates else None

    def save_policy_checkpoint(
        self,
        state_dict: dict,
        architecture: PolicyArchitecture | str,
        epoch: int,
        *,
        slot_id: str | None = None,
    ) -> str:
        checkpoint_root = self._checkpoint_root(slot_id)
        checkpoint_root.mkdir(parents=True, exist_ok=True)
        checkpoint_dir = (checkpoint_root / f"{self.run_name}:v{epoch}").expanduser().resolve()
        architecture_spec = architecture.to_spec() if isinstance(architecture, PolicyArchitecture) else architecture
        write_checkpoint_bundle(
            checkpoint_dir,
            architecture_spec=architecture_spec,
            state_dict=state_dict,
        )

        if self._remote_prefix:
            remote_base = self._checkpoint_uri_base(slot_id)
            remote_zip = f"{remote_base.rstrip('/')}/{checkpoint_dir.name}.zip"
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
                write_file(remote_zip, str(zip_path), content_type="application/zip")
            finally:
                zip_path.unlink(missing_ok=True)
            logger.debug("Policy checkpoint saved remotely to %s", remote_zip)
            return remote_zip

        logger.debug("Policy checkpoint saved locally to %s", checkpoint_dir.as_uri())
        return checkpoint_dir.as_uri()

    def load_trainer_state(self) -> Optional[Dict[str, Any]]:
        trainer_file = self.checkpoint_dir / "trainer_state.pt"
        if not trainer_file.exists():
            return None
        return torch.load(trainer_file, map_location="cpu", weights_only=False)

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
            torch.save(state, tmp_path)
            tmp_path.replace(self.checkpoint_dir / "trainer_state.pt")

        if is_schedulefree:
            optimizer.train()
