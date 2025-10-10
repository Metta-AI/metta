import logging
import os
import platform
import subprocess
import sys
from typing import Optional, TYPE_CHECKING

from pathlib import Path

from experiments.recipes.arena_basic_easy_shaped import (
    evaluate,
    evaluate_in_sweep,
    make_curriculum,
    mettagrid,
    play,
    replay,
    simulations,
    sweep_async_progressive,
    train as base_train,
)
from metta.agent.policy import PolicyArchitecture
from metta.cogworks.curriculum.curriculum import CurriculumConfig
from metta.rl.trainer_config import TorchProfilerConfig
from metta.tools.train import TrainTool

logger = logging.getLogger(__name__)

if TYPE_CHECKING:  # pragma: no cover
    pass


def _supports_mem_eff_path() -> bool:
    try:
        from causal_conv1d import causal_conv1d_fn  # type: ignore[attr-defined]
    except ModuleNotFoundError:
        return False

    return callable(causal_conv1d_fn)


DEFAULT_LEARNING_RATE = 8e-4
DEFAULT_BATCH_SIZE = 131_072
DEFAULT_MINIBATCH_SIZE = 4_096
DEFAULT_FORWARD_PASS_MINIBATCH_TARGET_SIZE = 1_024


def _apply_overrides(
    tool: TrainTool,
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
    tool.torch_profiler = TorchProfilerConfig(interval_epochs=0)


def _ensure_cuda_extras_installed() -> None:
    if platform.system() != "Linux":
        return

    script_path = (
        Path(__file__).resolve().parents[4] / "scripts" / "install_cuda_extras.py"
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
    curriculum: Optional[CurriculumConfig] = None,
    enable_detailed_slice_logging: bool = False,
    policy_architecture: PolicyArchitecture | None = None,
    learning_rate: float = DEFAULT_LEARNING_RATE,
    batch_size: int = DEFAULT_BATCH_SIZE,
    minibatch_size: int = DEFAULT_MINIBATCH_SIZE,
    forward_pass_minibatch_target_size: int = DEFAULT_FORWARD_PASS_MINIBATCH_TARGET_SIZE,
) -> TrainTool:
    _ensure_cuda_extras_installed()

    try:
        from metta.agent.policies.drama_policy import DramaPolicyConfig
        from metta.agent.components.drama import DramaWorldModelConfig
    except ModuleNotFoundError as exc:
        if exc.name == "mamba_ssm":
            raise RuntimeError(
                "DRAMA recipes require the `mamba-ssm` package (Linux + CUDA)."
                " Install it on a supported system before running this recipe."
            ) from exc
        raise

    policy = policy_architecture or DramaPolicyConfig()

    mem_eff_supported = _supports_mem_eff_path()
    if not mem_eff_supported:
        logger.warning(
            "[ABES Drama] Detected missing causal-conv1d CUDA kernels; disabling memory-efficient path."
            " Install `flash-attn` and `causal-conv1d` for best performance."
        )

    for component in policy.components:
        if isinstance(component, DramaWorldModelConfig):
            ssm_cfg = dict(component.ssm_cfg) if component.ssm_cfg else {}
            if not mem_eff_supported:
                ssm_cfg["use_mem_eff_path"] = False
            component.ssm_cfg = ssm_cfg

    tool = base_train(
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
    "sweep_async_progressive",
    "train",
]
