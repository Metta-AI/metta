from metta.common.tool import Tool
from pydantic import Field
from typing import Any, Optional

from metta.sweep.ray.ray_controller import SweepConfig, ray_sweep


class RaySweepTool(Tool):

    # This run is NOT used anywhere, but skypilot
    # always passes one. This is fine for development.
    run: Optional[str] = None

    sweep_config: SweepConfig = SweepConfig()
    search_space: dict[str, Any] = Field(default_factory=dict)

    # TODO This is unused at the moment
    ray_config: Optional[dict[str, Any]] = None

    #  TODO: Add some logging here?

    def invoke(self, args: dict[str, str]) -> int | None:
        return ray_sweep(
            sweep_config=self.sweep_config,
            search_space=self.search_space,
            ray_address=None
        )
