import logging
import os
import platform
import subprocess
import sys
import typing

import pathlib

import experiments.recipes.arena_basic_easy_shaped
import metta.agent.policy
import metta.cogworks.curriculum.curriculum
import metta.rl.trainer_config
import metta.tools.train

logger = logging.getLogger(__name__)

if typing.TYPE_CHECKING:  # pragma: no cover
    pass


def _supports_mem_eff_path() -> bool:
    try:
        import causal_conv1d  # type: ignore[attr-defined]
    except ModuleNotFoundError:
        return False

    return callable(causal_conv1d.causal_conv1d_fn)


DEFAULT_LEARNING_RATE = 8e-4
DEFAULT_BATCH_SIZE = 131_072
DEFAULT_MINIBATCH_SIZE = 4_096
DEFAULT_FORWARD_PASS_MINIBATCH_TARGET_SIZE = 1_024


def _apply_overrides(
    tool: metta.tools.train.TrainTool,
    *,
    learning_rate: float,
    batch_size: int,
    minibatch_size: int,
    forward_pass_minibatch_target_size: int,
) -> None:
    trainer = tool.trainer
    trainer.optimizer.learning_rate = learning_rate
    trainer.batch_size = batch_size
    trainer.minibatch_size = minibatch_size

    tool.training_env.forward_pass_minibatch_target_size = (
        forward_pass_minibatch_target_size
    )
    tool.torch_profiler = metta.rl.trainer_config.TorchProfilerConfig(interval_epochs=0)


def _ensure_cuda_extras_installed() -> None:
    if platform.system() != "Linux":
        return

    script_path = (
        pathlib.Path(__file__).resolve().parents[4]
        / "scripts"
        / "install_cuda_extras.py"
    )
    if not script_path.exists():
        logger.warning(
            "Could not locate install_cuda_extras.py; skipping CUDA extras installation."
        )
        return

    env = os.environ.copy()
    try:
        subprocess.run(
            [sys.executable, str(script_path), "--quiet"],
            check=True,
            env=env,
        )
    except subprocess.CalledProcessError as exc:
        logger.warning("Failed to install CUDA extras automatically: %s", exc)
    except Exception as exc:  # pragma: no cover - defensive logging
        logger.warning("Unexpected error while installing CUDA extras: %s", exc)


def train(
    *,
    curriculum: typing.Optional[
        metta.cogworks.curriculum.curriculum.CurriculumConfig
    ] = None,
    enable_detailed_slice_logging: bool = False,
    policy_architecture: metta.agent.policy.PolicyArchitecture | None = None,
    learning_rate: float = DEFAULT_LEARNING_RATE,
    batch_size: int = DEFAULT_BATCH_SIZE,
    minibatch_size: int = DEFAULT_MINIBATCH_SIZE,
    forward_pass_minibatch_target_size: int = DEFAULT_FORWARD_PASS_MINIBATCH_TARGET_SIZE,
) -> metta.tools.train.TrainTool:
    _ensure_cuda_extras_installed()

    try:
        import metta.agent.policies.drama_policy
        import metta.agent.components.drama
    except ModuleNotFoundError as exc:
        if exc.name == "mamba_ssm":
            raise RuntimeError(
                "DRAMA recipes require the `mamba-ssm` package (Linux + CUDA)."
                " Install it on a supported system before running this recipe."
            ) from exc
        raise

    policy = (
        policy_architecture or metta.agent.policies.drama_policy.DramaPolicyConfig()
    )

    mem_eff_supported = _supports_mem_eff_path()
    if not mem_eff_supported:
        logger.warning(
            "[ABES Drama] Detected missing causal-conv1d CUDA kernels; disabling memory-efficient path."
            " Install `flash-attn` and `causal-conv1d` for best performance."
        )

    for component in policy.components:
        if isinstance(component, metta.agent.components.drama.DramaWorldModelConfig):
            ssm_cfg = dict(component.ssm_cfg) if component.ssm_cfg else {}
            if not mem_eff_supported:
                ssm_cfg["use_mem_eff_path"] = False
            component.ssm_cfg = ssm_cfg

    tool = experiments.recipes.arena_basic_easy_shaped.train(
        curriculum=curriculum,
        enable_detailed_slice_logging=enable_detailed_slice_logging,
        policy_architecture=policy,
    )

    _apply_overrides(
        tool,
        learning_rate=learning_rate,
        batch_size=batch_size,
        minibatch_size=minibatch_size,
        forward_pass_minibatch_target_size=forward_pass_minibatch_target_size,
    )

    return tool


__all__ = [
    "mettagrid",
    "make_curriculum",
    "simulations",
    "play",
    "replay",
    "evaluate",
    "evaluate_in_sweep",
    "sweep",
    "train",
]


for _name in [
    "mettagrid",
    "make_curriculum",
    "simulations",
    "play",
    "replay",
    "evaluate",
    "evaluate_in_sweep",
    "sweep",
]:
    if _name in globals():
        continue
    globals()[_name] = getattr(experiments.recipes.arena_basic_easy_shaped, _name)

del _name
