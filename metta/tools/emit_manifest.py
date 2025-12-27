from __future__ import annotations

import logging
from pathlib import Path
from typing import Optional

from pydantic import Field

from metta.agent.policy import PolicyArchitecture
from metta.common.tool import Tool
from mettagrid.policy.mpt_artifact import load_mpt

logger = logging.getLogger(__name__)


class EmitManifestTool(Tool):
    policy_architecture: Optional[PolicyArchitecture] = None
    checkpoint_uri: Optional[str] = None
    output_path: Path = Field(default_factory=lambda: Path("./policy_manifest.txt"))

    def invoke(self, args: dict[str, str]) -> int | None:
        if (self.policy_architecture is None) == (self.checkpoint_uri is None):
            raise ValueError("Specify exactly one of policy_architecture or checkpoint_uri")

        if self.checkpoint_uri is not None:
            architecture = load_mpt(self.checkpoint_uri).architecture
        else:
            architecture = self.policy_architecture
            assert architecture is not None

        spec = architecture.to_spec()
        self.output_path.parent.mkdir(parents=True, exist_ok=True)
        self.output_path.write_text(spec)
        logger.info("Wrote policy manifest to %s", self.output_path)
        return 0
