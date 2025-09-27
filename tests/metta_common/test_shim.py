"""Regression tests for the ``metta.common`` compatibility shim."""

from __future__ import annotations

import inspect


def test_test_support_exports() -> None:
    import metta.common.test_support as test_support

    assert hasattr(test_support, "docker_client_fixture")
    assert hasattr(test_support, "pytest_collection_modifyitems")
    assert hasattr(test_support, "isolated_test_schema_uri")

    for name in [
        "docker_client_fixture",
        "pytest_collection_modifyitems",
        "isolated_test_schema_uri",
    ]:
        attr = getattr(test_support, name)
        assert callable(attr), name


def test_key_util_modules_are_available() -> None:
    from metta.common.util import collections, constants, fs, log_config

    assert hasattr(constants, "METTA_WANDB_ENTITY")
    assert hasattr(constants, "SOFTMAX_S3_BUCKET")
    assert hasattr(collections, "group_by")
    assert hasattr(fs, "get_repo_root")
    assert hasattr(log_config, "init_logging")


def test_tool_and_wandb_modules_are_available() -> None:
    from metta.common.tool import Tool
    from metta.common.wandb import context, utils

    assert inspect.isclass(Tool)
    assert hasattr(context, "WandbRun")
    assert hasattr(context, "WandbContext")
    assert hasattr(utils, "abort_requested")


def test_silence_warnings_module_is_present() -> None:
    import metta.common.silence_warnings as silence_warnings

    # The module exists and is executed for its side effects when imported.
    assert silence_warnings.__name__ == "metta.common.silence_warnings"
    assert silence_warnings.__file__.endswith("silence_warnings.py")
