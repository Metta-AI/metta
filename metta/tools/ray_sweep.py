import os
from typing import Any, Optional

from pydantic import ConfigDict, Field

from metta.common.tool import Tool
from metta.sweep.ray.ray_controller import SweepConfig, ray_sweep

try:
    from ray.tune.search.sample import Categorical as _RayCategorical
    from ray.tune.search.sample import Domain as _RayDomain
    from ray.tune.search.sample import Float as _RayFloat
    from ray.tune.search.sample import Integer as _RayInteger

    # TODO I don't think we need any of this stuff

    _KNOWN_RAY_DOMAINS = {_RayDomain, _RayFloat, _RayInteger, _RayCategorical}
except (ModuleNotFoundError, ImportError):  # pragma: no cover - ray not available in all environments

    class _RayDomain:  # type: ignore[too-many-ancestors]
        """Fallback placeholder when Ray isn't installed."""

        pass

    _KNOWN_RAY_DOMAINS = {_RayDomain}


def _serialize_ray_domain(value: Any) -> str:
    return repr(value)


class RaySweepTool(Tool):
    model_config = ConfigDict(
        extra="forbid",
        arbitrary_types_allowed=True,
        json_encoders={domain_type: _serialize_ray_domain for domain_type in _KNOWN_RAY_DOMAINS},
    )

    # This run is NOT used anywhere, but skypilot
    # always passes one. This is fine for development.
    run: Optional[str] = None

    sweep_config: SweepConfig = SweepConfig()
    search_space: dict[str, Any] = Field(default_factory=dict)

    #  TODO: Add some logging here?

    def invoke(self, args: dict[str, str]) -> int | None:
        ray_address = args.get("ray_address") or os.getenv("RAY_ADDRESS")
        return ray_sweep(
            sweep_config=self.sweep_config,
            search_space=self.search_space,
            ray_address=ray_address,
        )
