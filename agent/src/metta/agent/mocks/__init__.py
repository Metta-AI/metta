from __future__ import annotations

import importlib

__all__ = ["mock_agent", "mock_policy", "MockAgent", "MockPolicy"]

_MODULES = {
    "mock_agent": "metta.agent.mocks.mock_agent",
    "mock_policy": "metta.agent.mocks.mock_policy",
}

_CLASS_EXPORTS = {
    "MockAgent": ("metta.agent.mocks.mock_agent", "MockAgent"),
    "MockPolicy": ("metta.agent.mocks.mock_policy", "MockPolicy"),
}


def __getattr__(name: str):
    module_path = _MODULES.get(name)
    if module_path is not None:
        module = importlib.import_module(module_path)
        globals()[name] = module
        return module

    class_target = _CLASS_EXPORTS.get(name)
    if class_target is not None:
        module_path, attr = class_target
        module = importlib.import_module(module_path)
        value = getattr(module, attr)
        globals()[name] = value
        return value

    raise AttributeError(f"module '{__name__}' has no attribute '{name}'")
