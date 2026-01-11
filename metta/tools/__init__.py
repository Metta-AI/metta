from __future__ import annotations

from importlib import import_module
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from metta.tools.eval import EvalWithResultTool, EvaluateTool
    from metta.tools.play import PlayTool
    from metta.tools.replay import ReplayTool
    from metta.tools.request_remote_eval import RequestRemoteEvalTool
    from metta.tools.resolve_uri import ResolveUriTool
    from metta.tools.stub import StubTool
    from metta.tools.sweep import SweepTool
    from metta.tools.train import TrainTool

_TOOL_MODULES = {
    "EvalWithResultTool": "metta.tools.eval",
    "EvaluateTool": "metta.tools.eval",
    "PlayTool": "metta.tools.play",
    "ReplayTool": "metta.tools.replay",
    "RequestRemoteEvalTool": "metta.tools.request_remote_eval",
    "ResolveUriTool": "metta.tools.resolve_uri",
    "StubTool": "metta.tools.stub",
    "SweepTool": "metta.tools.sweep",
    "TrainTool": "metta.tools.train",
}


def __getattr__(name: str) -> object:
    module_path = _TOOL_MODULES.get(name)
    if module_path is None:
        raise AttributeError(f\"module {__name__!r} has no attribute {name!r}\")
    module = import_module(module_path)
    return getattr(module, name)


def __dir__() -> list[str]:
    return sorted(set(globals()) | set(_TOOL_MODULES))

__all__ = [
    "EvalWithResultTool",
    "EvaluateTool",
    "PlayTool",
    "ReplayTool",
    "RequestRemoteEvalTool",
    "ResolveUriTool",
    "StubTool",
    "SweepTool",
    "TrainTool",
]
