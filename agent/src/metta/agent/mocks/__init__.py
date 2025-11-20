from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from metta.agent.mocks.mock_agent import MockAgent
    from metta.agent.mocks.mock_policy import MockPolicy

__all__ = ["MockPolicy", "MockAgent"]


def __getattr__(name: str):
    """Lazily import mock classes to avoid loading torch at package import."""
    if name == "MockAgent":
        from metta.agent.mocks.mock_agent import MockAgent

        globals()["MockAgent"] = MockAgent
        return MockAgent
    elif name == "MockPolicy":
        from metta.agent.mocks.mock_policy import MockPolicy

        globals()["MockPolicy"] = MockPolicy
        return MockPolicy
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
