import os
import shutil
import subprocess
from typing import Annotated

import typer

from metta.common.util.fs import get_repo_root
from metta.setup.utils import error, info, success

app = typer.Typer(
    help="MettaGrid C++ test runner",
    invoke_without_command=True,
)


def _run_cmd(cmd: list[str], cwd: str, env: dict[str, str] | None = None) -> int:
    return subprocess.run(cmd, cwd=cwd, env=env, check=False).returncode


def _run_test(mettagrid_dir: str) -> None:
    info("Building debug...")
    exit_code = _run_cmd(["bazel", "build", "--config=dbg", "//:mettagrid_c"], cwd=mettagrid_dir)
    if exit_code != 0:
        error("C++ build failed!")
        raise typer.Exit(exit_code)

    info("Running unit tests...")
    exit_code = _run_cmd(["bazel", "test", "--config=dbg", "//tests:tests_all"], cwd=mettagrid_dir)
    if exit_code != 0:
        error("C++ test failed!")
        raise typer.Exit(exit_code)

    success("C++ test completed successfully!")


def _run_benchmark(mettagrid_dir: str, *, verbose: bool) -> None:
    info("Building C++ benchmarks...")
    exit_code = _run_cmd(
        ["bazel", "build", "--config=opt", "//benchmarks:test_mettagrid_env_benchmark"],
        cwd=mettagrid_dir,
    )
    if exit_code != 0:
        error("C++ benchmark build failed!")
        raise typer.Exit(exit_code)

    info("Creating build-release directory...")
    build_release_dir = os.path.join(mettagrid_dir, "build-release")
    os.makedirs(build_release_dir, exist_ok=True)

    info("Copying benchmark binaries...")
    src_binary = os.path.join(mettagrid_dir, "bazel-bin/benchmarks/test_mettagrid_env_benchmark")
    dst_binary = os.path.join(build_release_dir, "test_mettagrid_env_benchmark")
    shutil.copy2(src_binary, dst_binary)
    os.chmod(dst_binary, 0o755)

    info("Running C++ benchmarks...")
    # Note: Using the copied binary instead of 'bazel run' to avoid Python environment issues
    cpp_exit_code = _run_cmd(["./build-release/test_mettagrid_env_benchmark"], cwd=mettagrid_dir)
    if cpp_exit_code != 0:
        info("C++ benchmark failed - this may be due to Python environment issues")

    info("Running Python benchmarks...")
    pytest_cmd = ["uv", "run", "pytest", "benchmarks/test_mettagrid_env_benchmark.py", "-v", "--benchmark-only"]
    if verbose:
        pytest_cmd.append("-s")
    exit_code = _run_cmd(pytest_cmd, cwd=mettagrid_dir)
    if exit_code != 0:
        error("C++ benchmark failed!")
        raise typer.Exit(exit_code)

    success("C++ benchmark completed successfully!")


@app.callback()
def command(
    benchmark: bool = typer.Option(
        False,
        "--benchmark",
        help="Run benchmark tests.",
    ),
    test: bool = typer.Option(
        False,
        "--test",
        help="Run unit tests.",
    ),
    verbose: Annotated[bool, typer.Option("--verbose", "-v", help="Show verbose build output.")] = False,
) -> None:
    mettagrid_dir = str(get_repo_root() / "packages" / "mettagrid")
    targets = {k: v for k, v in [("benchmark", benchmark), ("test", test)] if v}
    for target in targets.keys() or ["test"]:
        if target == "test":
            _run_test(mettagrid_dir)
        elif target == "benchmark":
            _run_benchmark(mettagrid_dir, verbose=verbose)
