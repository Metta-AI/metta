"""Lightweight stub tool useful for exercising the tool runner."""

from __future__ import annotations

import logging
from typing import Any

from pydantic import ConfigDict, Field

from metta.common.tool import Tool

logger = logging.getLogger(__name__)


class StubTool(Tool):
    """Minimal tool implementation that happily accepts arbitrary config."""

    model_config = ConfigDict(extra="allow")

    exit_code: int = Field(default=0, description="Exit code returned when invoke completes.")
    log_arguments: bool = Field(default=True, description="Log accepted config and CLI args when invoking.")

    def _extra_payload(self) -> dict[str, Any]:
        """Return any dynamically provided config that is not part of Tool."""
        extra = getattr(self, "model_extra", None)
        if not extra:
            return {}
        return dict(extra)

    def invoke(self, args: dict[str, str]) -> int | None:
        """Log provided arguments and exit with the configured status code."""
        if self.log_arguments:
            dynamic_config = self._extra_payload()
            if dynamic_config:
                logger.info("StubTool dynamic config: %s", dynamic_config)
            else:
                logger.info("StubTool invoked without dynamic config overrides.")
            if args:
                logger.info("StubTool CLI args: %s", args)
        return self.exit_code
