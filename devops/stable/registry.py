from __future__ import annotations

import importlib
import pkgutil
from dataclasses import dataclass, field
from enum import StrEnum
from typing import Callable, get_type_hints

from devops.stable.runner import AcceptanceCriterion, Job
from metta.common.tool import Tool
from metta.common.wandb.context import WandbConfig
from metta.tools.train import TrainTool


class Suite(StrEnum):
    CI = "ci"
    STABLE = "stable"


@dataclass
class JobSpec:
    """Specification for a registered job."""

    func: Callable[..., Tool]
    suite: Suite
    depends_on: Callable[..., Tool] | None = None
    input_references: dict[str, str] = field(default_factory=dict)
    timeout_s: int = 3600

    # Only used for remote jobs
    remote_gpus: int | None = None
    remote_nodes: int | None = None

    acceptance: list[AcceptanceCriterion] = field(default_factory=list)

    @property
    def name(self) -> str:
        module = self.func.__module__
        short_module = module.replace("recipes.prod.", "").replace("recipes.experiment.", "")
        return f"{short_module}.{self.func.__name__}"


_registry: list[JobSpec] = []


def ci_job(
    *,
    depends_on: Callable[..., Tool] | None = None,
    input_references: dict[str, str] | None = None,
    timeout_s: int = 300,
) -> Callable[[Callable[..., Tool]], Callable[..., Tool]]:
    """Register a CI job.

    Args:
        depends_on: Function this job depends on (must complete first).
        input_references: Map of parameter names to output fields from dependency.
        timeout_s: Maximum time for job to complete (default 5min for CI).
    """

    def decorator(func: Callable[..., Tool]) -> Callable[..., Tool]:
        _registry.append(
            JobSpec(
                func=func,
                suite=Suite.CI,
                depends_on=depends_on,
                input_references=input_references or {},
                timeout_s=timeout_s,
            )
        )
        return func

    return decorator


def stable_job(
    *,
    depends_on: Callable[..., Tool] | None = None,
    input_references: dict[str, str] | None = None,
    timeout_s: int = 7200,
    remote_gpus: int | None = None,
    remote_nodes: int | None = None,
    acceptance: list[AcceptanceCriterion] | None = None,
) -> Callable[[Callable[..., Tool]], Callable[..., Tool]]:
    """Register a stable release job.

    Args:
        depends_on: Function this job depends on (must complete first).
        inject: Map of parameter names to output fields from dependency.
        timeout_s: Maximum time for job to complete (default 2h for stable).
        remote_gpus: Number of GPUs (None = local, 1+ = remote SkyPilot).
        remote_nodes: Number of nodes for multi-node training.
        acceptance: List of acceptance criteria to evaluate after job completes.
    """

    def decorator(func: Callable[..., Tool]) -> Callable[..., Tool]:
        _registry.append(
            JobSpec(
                func=func,
                suite=Suite.STABLE,
                depends_on=depends_on,
                input_references=input_references or {},
                timeout_s=timeout_s,
                remote_gpus=remote_gpus,
                remote_nodes=remote_nodes,
                acceptance=acceptance or [],
            )
        )
        return func

    return decorator


def discover_jobs(suite: Suite | None = None) -> list[JobSpec]:
    """Discover all registered jobs by importing recipes.prod and recipes.experiment.

    Args:
        suite: Filter by suite, or None to return all jobs.

    Returns:
        List of JobSpec objects for matching jobs.
    """
    _registry.clear()

    import recipes.experiment as experiment_package
    import recipes.prod as prod_package

    for package, prefix in [
        (prod_package, "recipes.prod."),
        (experiment_package, "recipes.experiment."),
    ]:
        for module_info in pkgutil.walk_packages(package.__path__, prefix=prefix):
            try:
                importlib.import_module(module_info.name)
            except Exception:
                pass

    if suite is None:
        return list(_registry)
    return [spec for spec in _registry if spec.suite == suite]


def specs_to_jobs(specs: list[JobSpec], prefix: str) -> list[Job]:
    """Convert job specs to Job objects for the runner.

    Args:
        specs: Job specifications from the registry.
        prefix: Prefix for job names (e.g., "runner.20251210").

    Returns:
        List of Job objects ready for the runner.
    """
    jobs: list[Job] = []

    spec_to_job_name: dict[Callable, str] = {}
    for spec in specs:
        spec_to_job_name[spec.func] = f"{prefix}-{spec.name}"

    for spec in specs:
        tool_path = f"{spec.func.__module__}.{spec.func.__name__}"
        job_name = spec_to_job_name[spec.func]

        return_type = get_type_hints(spec.func).get("return")

        wandb_disabled = True
        if return_type is TrainTool:
            tool = spec.func()
            assert isinstance(tool, TrainTool)
            wandb_disabled = not (tool.wandb.enabled or tool.wandb == WandbConfig.Unconfigured())

        if spec.acceptance and wandb_disabled:
            raise ValueError(f"{spec.name} must have wandb enabled to use acceptance criteria")

        assert (spec.remote_gpus is not None) == (spec.remote_nodes is not None), (
            "remote jobs must have either gpus or nodes"
        )
        if spec.remote_gpus:
            cmd = [
                "uv",
                "run",
                "./devops/skypilot/launch.py",
                tool_path,
                f"--gpus={spec.remote_gpus}",
                f"--nodes={spec.remote_nodes}",
                "--skip-git-check",
            ]
        else:
            cmd = ["uv", "run", "./tools/run.py", tool_path]
        if return_type is TrainTool:
            cmd.append(f"run={job_name}")

        dependencies: list[str] = []
        if spec.depends_on and spec.depends_on in spec_to_job_name:
            dep_job_name = spec_to_job_name[spec.depends_on]
            dependencies.append(dep_job_name)

            inject_args = []
            available_references = spec.depends_on().output_references(job_name=dep_job_name)
            for param_name, output_field in spec.input_references.items():
                if output_field not in available_references:
                    raise ValueError(f"Dependency {dep_job_name} does not provide output field: {output_field}")
                value = available_references[output_field]
                inject_args.append(f"{param_name}={value}")
            cmd.extend(inject_args)

        jobs.append(
            Job(
                name=job_name,
                cmd=cmd,
                timeout_s=spec.timeout_s,
                remote_gpus=spec.remote_gpus,
                remote_nodes=spec.remote_nodes,
                dependencies=dependencies,
                wandb_run_name=job_name if not wandb_disabled else None,
                acceptance=spec.acceptance,
            )
        )

    return jobs
