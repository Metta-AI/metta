from __future__ import annotations

import importlib
import sys
from pathlib import Path
from typing import Any

import torch
import torch.nn as nn
from transformers import PreTrainedModel

from .configuration_metta_policy import MettaPolicyConfig


class MettaPolicyForRL(PreTrainedModel):
    """Wrapper that rehydrates a Metta policy saved in Hugging Face format."""

    config_class = MettaPolicyConfig

    def __init__(self, config: MettaPolicyConfig, *model_args: Any, **model_kwargs: Any) -> None:
        super().__init__(config)
        artifact_root = getattr(config, "_artifact_root", None)
        if artifact_root is None:
            artifact_root = Path(__file__).resolve().parent.parent
        else:
            artifact_root = Path(artifact_root)
        self._root_dir = artifact_root
        self._ensure_snapshot_on_path()
        policy = self._load_policy()
        if not isinstance(policy, nn.Module):
            raise TypeError(f"Expected checkpoint to contain an nn.Module, found {type(policy)!r}")
        self.policy = policy
        self.policy.eval()
        self.post_init()

    @classmethod
    def from_pretrained(
        cls,
        pretrained_model_name_or_path: str | Path,
        *model_args: Any,
        **kwargs: Any,
    ) -> "MettaPolicyForRL":
        config: MettaPolicyConfig | None = kwargs.pop("config", None)
        if config is None:
            config = cls.config_class.from_pretrained(pretrained_model_name_or_path)  # type: ignore[arg-type]
        config._artifact_root = Path(pretrained_model_name_or_path)
        model = cls(config, *model_args, **kwargs)
        return model

    def forward(self, *args: Any, **kwargs: Any) -> Any:  # type: ignore[override]
        return self.policy(*args, **kwargs)

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------
    def _ensure_snapshot_on_path(self) -> None:
        snapshot_dir = self._root_dir / self.config.code_snapshot_subdir
        if snapshot_dir.exists():
            snapshot_path = str(snapshot_dir)
            if snapshot_path not in sys.path:
                sys.path.insert(0, snapshot_path)
        for relative in reversed(self.config.code_roots):
            candidate = (snapshot_dir / relative).resolve()
            if candidate.exists():
                candidate_path = str(candidate)
                if candidate_path not in sys.path:
                    sys.path.insert(0, candidate_path)
        for module_name in ("metta", "metta.agent"):
            if module_name in sys.modules:
                del sys.modules[module_name]
        importlib.invalidate_caches()
        try:
            importlib.import_module("metta")
        except ModuleNotFoundError:
            pass

    def _load_policy(self) -> nn.Module:
        checkpoint_path = self._root_dir / self.config.checkpoint_filename
        if not checkpoint_path.exists():
            raise FileNotFoundError(f"Checkpoint file not found: {checkpoint_path}")
        policy = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
        return policy


def ensure_module_available(module_name: str) -> None:
    """Import a module, raising a clear error if it cannot be loaded from the snapshot."""

    try:
        importlib.import_module(module_name)
    except ModuleNotFoundError as exc:  # pragma: no cover - defensive guard
        raise ModuleNotFoundError(
            f"Module '{module_name}' could not be imported from the bundled snapshot."
        ) from exc
