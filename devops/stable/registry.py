from __future__ import annotations

import importlib
import pkgutil
from dataclasses import dataclass, field
from enum import StrEnum
from typing import Callable, get_type_hints

from devops.stable.runner import AcceptanceCriterion, Job
from metta.common.tool import Tool
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
    inject: dict[str, str] = field(default_factory=dict)
    timeout_s: int = 3600
    gpus: int | None = None
    nodes: int = 1
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
    inject: dict[str, str] | None = None,
    timeout_s: int = 300,
    gpus: int | None = None,
    nodes: int = 1,
    acceptance: list[AcceptanceCriterion] | None = None,
) -> Callable[[Callable[..., Tool]], Callable[..., Tool]]:
    """Register a CI job.

    Args:
        depends_on: Function this job depends on (must complete first).
        inject: Map of parameter names to output fields from dependency.
        timeout_s: Maximum time for job to complete (default 5min for CI).
        gpus: Number of GPUs (None = local, 1+ = remote SkyPilot).
        nodes: Number of nodes for multi-node training.
        acceptance: List of acceptance criteria to evaluate after job completes.
    """

    def decorator(func: Callable[..., Tool]) -> Callable[..., Tool]:
        _registry.append(
            JobSpec(
                func=func,
                suite=Suite.CI,
                depends_on=depends_on,
                inject=inject or {},
                timeout_s=timeout_s,
                gpus=gpus,
                nodes=nodes,
                acceptance=acceptance or [],
            )
        )
        return func

    return decorator


def stable_job(
    *,
    depends_on: Callable[..., Tool] | None = None,
    inject: dict[str, str] | None = None,
    timeout_s: int = 7200,
    gpus: int | None = None,
    nodes: int = 1,
    acceptance: list[AcceptanceCriterion] | None = None,
) -> Callable[[Callable[..., Tool]], Callable[..., Tool]]:
    """Register a stable release job.

    Args:
        depends_on: Function this job depends on (must complete first).
        inject: Map of parameter names to output fields from dependency.
        timeout_s: Maximum time for job to complete (default 2h for stable).
        gpus: Number of GPUs (None = local, 1+ = remote SkyPilot).
        nodes: Number of nodes for multi-node training.
        acceptance: List of acceptance criteria to evaluate after job completes.
    """

    def decorator(func: Callable[..., Tool]) -> Callable[..., Tool]:
        _registry.append(
            JobSpec(
                func=func,
                suite=Suite.STABLE,
                depends_on=depends_on,
                inject=inject or {},
                timeout_s=timeout_s,
                gpus=gpus,
                nodes=nodes,
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
        short_name = spec.name.replace(".", "_")
        spec_to_job_name[spec.func] = f"{prefix}.{short_name}"

    for spec in specs:
        tool_path = f"{spec.func.__module__}.{spec.func.__name__}"
        job_name = spec_to_job_name[spec.func]

        return_type = get_type_hints(spec.func).get("return")
        is_train = return_type is TrainTool
        is_remote = spec.gpus is not None

        wandb_enabled = False
        if is_train:
            tool = spec.func()
            wandb_enabled = tool.wandb.enabled

        if spec.acceptance and not wandb_enabled:
            raise ValueError(f"{spec.name} has acceptance criteria but wandb is disabled")

        if is_remote:
            cmd = [
                "uv",
                "run",
                "./devops/skypilot/launch.py",
                tool_path,
                f"--gpus={spec.gpus}",
                f"--nodes={spec.nodes}",
                "--skip-git-check",
            ]
        else:
            cmd = ["uv", "run", "./tools/run.py", tool_path]
        if is_train:
            cmd.append(f"run={job_name}")

        dependencies: list[str] = []
        if spec.depends_on and spec.depends_on in spec_to_job_name:
            dep_job_name = spec_to_job_name[spec.depends_on]
            dependencies.append(dep_job_name)

            inject_args = []
            for param_name, output_field in spec.inject.items():
                if output_field == "uri":
                    dep_tool = spec.depends_on()
                    if not isinstance(dep_tool, TrainTool):
                        raise ValueError(
                            f"Only support injecting 'uri' from TrainTool, but {dep_job_name} is {type(dep_tool)}"
                        )
                    value = dep_tool.output_uri(dep_job_name)
                    inject_args.append(f"{param_name}={value}")
                else:
                    raise ValueError(f"Unknown output field: {output_field}")

            cmd.extend(inject_args)

        remote = {"gpus": spec.gpus, "nodes": spec.nodes} if spec.gpus else None

        jobs.append(
            Job(
                name=job_name,
                cmd=cmd,
                timeout_s=spec.timeout_s,
                remote=remote,
                dependencies=dependencies,
                wandb_run_name=job_name if wandb_enabled else None,
                acceptance=spec.acceptance,
            )
        )

    return jobs
