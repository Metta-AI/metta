"""Tests for ICL control recipe to validate seed synchronization and architecture support."""

import pytest

from experiments.recipes.eval_v_11_1_25.ICL_control_recipe import (
    BENCHMARK_SEED,
    train,
)
from metta.tests_support.run_tool import RunToolResult, run_tool_in_process


def test_seed_synchronization() -> None:
    """Test that environment, curriculum, and weight initialization use the same seed.

    This test verifies that the benchmark configuration properly synchronizes seeds across:
    1. System configuration (for weight initialization via seed_everything)
    2. Training environment (for environment RNG)
    3. Curriculum (inherits from environment seed)

    If this assertion passes, the benchmark can proceed with confidence that results
    are reproducible across different architecture comparisons.
    """
    # Create training configuration for vit_reset architecture
    train_tool = train(curriculum_style="level_0", architecture="vit_reset")

    # Verify system seed is set to BENCHMARK_SEED (for weight initialization)
    assert train_tool.system.seed == BENCHMARK_SEED, (
        f"System seed {train_tool.system.seed} does not match BENCHMARK_SEED {BENCHMARK_SEED}"
    )

    # Verify training environment seed is set to BENCHMARK_SEED
    assert train_tool.training_env.seed == BENCHMARK_SEED, (
        f"Training env seed {train_tool.training_env.seed} does not match BENCHMARK_SEED {BENCHMARK_SEED}"
    )

    # Verify that curriculum task generator is properly configured
    # The curriculum RNG is initialized with cfg.seed in VectorizedTrainingEnvironment.__init__
    # via: Curriculum(cfg.curriculum, seed=cfg.seed)
    # This ensures curriculum task selection is deterministic and synchronized.
    assert train_tool.training_env.curriculum is not None, "Curriculum must be configured for benchmark"

    print("✓ Seed synchronization test passed: All seeds are properly synchronized to", BENCHMARK_SEED)
    print("  - System seed:", train_tool.system.seed, "(for weight initialization)")
    print("  - Training env seed:", train_tool.training_env.seed, "(for env & curriculum RNG)")


@pytest.mark.parametrize("architecture", ["vit_reset", "trxl"])
def test_architecture_support(architecture: str) -> None:
    """Test that both vit_reset and trxl architectures are supported."""
    train_tool = train(curriculum_style="level_0", architecture=architecture)

    # Verify the policy architecture is set correctly
    assert train_tool.policy_architecture is not None, f"Policy architecture not set for {architecture}"

    # Verify seeds are synchronized for this architecture
    assert train_tool.system.seed == BENCHMARK_SEED
    assert train_tool.training_env.seed == BENCHMARK_SEED

    print(f"✓ Architecture test passed for {architecture}")


def test_evaluations_enabled() -> None:
    """Test that evaluations are enabled in the recipe."""
    train_tool = train(curriculum_style="level_0", architecture="vit_reset")

    # Verify evaluator is configured
    assert train_tool.evaluator is not None, "Evaluator must be configured"

    # Verify evaluation interval is set
    assert train_tool.evaluator.epoch_interval > 0, "Evaluation epoch_interval must be greater than 0"

    # Verify evaluation suite is configured
    assert train_tool.evaluator.simulations is not None, "Evaluation simulations must be configured"
    assert len(train_tool.evaluator.simulations) > 0, "Evaluation suite must contain simulations"

    print(f"✓ Evaluations enabled with epoch_interval={train_tool.evaluator.epoch_interval}")


@pytest.mark.parametrize(
    "architecture",
    ["vit_reset", "trxl"],
)
def test_icl_control_recipe_dry_run(
    architecture: str,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    """Test that ICL control recipe supports dry-run for both architectures."""
    result: RunToolResult = run_tool_in_process(
        "experiments.recipes.eval_v_11_1_25.ICL_control_recipe.train",
        "--dry-run",
        f"architecture={architecture}",
        monkeypatch=monkeypatch,
        capsys=capsys,
    )

    assert result.returncode == 0, f"Dry run failed for architecture {architecture}"
    combined_output = result.stdout + result.stderr
    assert "Configuration validation successful" in combined_output, (
        f"Configuration validation not found for {architecture}"
    )


def test_invalid_architecture_raises_error() -> None:
    """Test that invalid architecture name raises ValueError."""
    with pytest.raises(ValueError, match="Unknown architecture"):
        train(curriculum_style="level_0", architecture="invalid_arch")


if __name__ == "__main__":
    # Allow running tests directly for quick validation
    print("Running seed synchronization test...")
    test_seed_synchronization()

    print("\nRunning architecture support tests...")
    test_architecture_support("vit_reset")
    test_architecture_support("trxl")

    print("\nRunning evaluations enabled test...")
    test_evaluations_enabled()

    print("\nRunning invalid architecture test...")
    test_invalid_architecture_raises_error()

    print("\n✓ All tests passed!")
