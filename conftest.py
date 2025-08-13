import contextlib
import os
import sys
import types

import pytest

from metta.common.test_support import docker_client_fixture


def pytest_configure(config):
    # Stub torch._dynamo and polyfills to avoid import-time errors when torch.compile internals are missing
    if "torch._dynamo" not in sys.modules:
        sys.modules["torch._dynamo"] = types.ModuleType("torch._dynamo")
    # Provide minimal attributes used by torch._compile wrapper
    dyn = sys.modules["torch._dynamo"]
    if not hasattr(dyn, "disable"):

        def _disable(fn=None, *a, **k):
            # Support both @torch._dynamo.disable and @torch._dynamo.disable()
            if fn is None:

                def _decorator(inner):
                    return inner

                return _decorator
            return fn

        dyn.disable = _disable
    if not hasattr(dyn, "reset"):
        dyn.reset = lambda *a, **k: None
    if not hasattr(dyn, "config"):
        dyn.config = types.SimpleNamespace(cache_size_limit=0)
    if not hasattr(dyn, "assume_constant_result"):
        dyn.assume_constant_result = lambda fn: fn
    if not hasattr(dyn, "graph_break"):
        dyn.graph_break = lambda *a, **k: None
    if "torch._dynamo.polyfills" not in sys.modules:
        sys.modules["torch._dynamo.polyfills"] = types.ModuleType("torch._dynamo.polyfills")
    if "torch._dynamo.polyfills.loader" not in sys.modules:
        sys.modules["torch._dynamo.polyfills.loader"] = types.ModuleType("torch._dynamo.polyfills.loader")
    # Disable torch.compile / Dynamo to avoid CPU AMP issues in tests
    os.environ.setdefault("TORCH_COMPILE_DISABLE", "1")
    os.environ.setdefault("TORCHDYNAMO_DISABLE", "1")
    os.environ.setdefault("PYTORCH_JIT_DISABLE", "1")

    # Pre-import torch and stub missing CPU AMP autocast to avoid import-time crashes
    try:
        import torch

        # Ensure torch.cpu.amp.autocast_mode.autocast exists on platforms without CPU AMP
        cpu_amp = getattr(torch.cpu, "amp", None)
        if cpu_amp is None:
            torch.cpu.amp = types.SimpleNamespace()  # type: ignore[attr-defined]
            cpu_amp = torch.cpu.amp  # type: ignore[attr-defined]
        if not hasattr(cpu_amp, "autocast_mode"):
            cpu_amp.autocast_mode = types.SimpleNamespace()  # type: ignore[attr-defined]
        if not hasattr(cpu_amp.autocast_mode, "autocast"):
            cpu_amp.autocast_mode.autocast = contextlib.nullcontext  # type: ignore[attr-defined]
    except Exception:
        pass

    # Add multiple markers correctly
    config.addinivalue_line("markers", "benchmark: mark a test as a benchmark test")
    config.addinivalue_line("markers", "verbose: mark a test to display verbose output")
    config.addinivalue_line("markers", "slow: mark a test as slow (runs in second phase)")


@pytest.fixture
def verbose(request):
    """Fixture that can be used in tests to check if verbose mode is enabled."""
    marker = request.node.get_closest_marker("verbose")
    return marker is not None


# Properly handle output capture for verbose tests
@pytest.hookimpl(hookwrapper=True)
def pytest_runtest_makereport(item, call):
    outcome = yield
    report = outcome.get_result()

    # Only process after the call phase (actual test execution)
    if report.when == "call" and item.get_closest_marker("verbose"):
        capman = item.config.pluginmanager.get_plugin("capturemanager")
        if capman and hasattr(report, "capstdout") and hasattr(report, "capstderr"):
            # Print the captured output with formatting
            print(f"\n\n===== VERBOSE OUTPUT FOR: {item.name} =====\n")
            if report.capstdout:
                print("--- STDOUT ---")
                print(report.capstdout)
            if report.capstderr:
                print("--- STDERR ---")
                print(report.capstderr)
            print(f"===== END VERBOSE OUTPUT FOR: {item.name} =====\n")


docker_client = docker_client_fixture()


@pytest.fixture(autouse=True, scope="session")
def _patch_torch_compile():
    """Ensure torch.compile is a no-op during tests.

    Some transitive imports may trigger torch._dynamo; make compile a no-op to
    avoid importing unsupported CPU AMP contexts on macOS.
    """
    try:
        import torch

        # Best-effort disable of dynamo if available
        if hasattr(torch, "_dynamo"):
            try:
                torch._dynamo.reset()  # type: ignore[attr-defined]
                torch._dynamo.config.cache_size_limit = 0  # type: ignore[attr-defined]
                torch._dynamo.disable()  # type: ignore[attr-defined]
            except Exception:
                pass

        # Make torch.compile a no-op
        if hasattr(torch, "compile"):

            def _noop_compile(fn=None, *args, **kwargs):
                return fn

            torch.compile = _noop_compile  # type: ignore[assignment]
    except Exception:
        # If torch isn't available or import fails, just skip patching
        pass
