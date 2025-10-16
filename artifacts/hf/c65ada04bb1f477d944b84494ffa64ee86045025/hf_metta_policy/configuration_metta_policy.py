from __future__ import annotations

from typing import Any, Dict, List, Optional

from transformers import PretrainedConfig


class MettaPolicyConfig(PretrainedConfig):
    """Configuration for Metta policies exported in Hugging Face format."""

    model_type = "metta-policy"

    def __init__(
        self,
        *,
        checkpoint_filename: str = "policy.pt",
        code_snapshot_subdir: str = "code_snapshot",
        source_commit: Optional[str] = None,
        extra_metadata: Optional[Dict[str, Any]] = None,
        code_roots: Optional[List[str]] = None,
        **kwargs: Any,
    ) -> None:
        super().__init__(**kwargs)
        self.checkpoint_filename = checkpoint_filename
        self.code_snapshot_subdir = code_snapshot_subdir
        self.source_commit = source_commit
        self.extra_metadata = extra_metadata or {}
        self.code_roots = code_roots or []
        self.auto_map = {
            "AutoConfig": ["hf_metta_policy.configuration_metta_policy", "MettaPolicyConfig"],
            "AutoModel": ["hf_metta_policy.modeling_metta_policy", "MettaPolicyForRL"],
        }
        self.architectures = ["MettaPolicyForRL"]
