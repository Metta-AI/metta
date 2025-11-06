"""Namespace package for cogworks subsystems."""

import importlib

__all__ = ["curriculum"]


def __getattr__(name: str):
    if name == "curriculum":
        module = importlib.import_module("metta.cogworks.curriculum")
        globals()[name] = module
        return module
    raise AttributeError(f"module 'metta.cogworks' has no attribute '{name}'")
