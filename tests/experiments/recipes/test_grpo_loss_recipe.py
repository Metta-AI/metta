import pytest

import metta.tests_support.run_tool


@pytest.mark.parametrize(
    "tool_path",
    [
        "experiments.recipes.losses.grpo.train",
        "experiments.recipes.losses.grpo.train_shaped",
        "experiments.recipes.losses.grpo.basic_easy_shaped",
    ],
)
def test_grpo_recipes_support_dry_run(
    tool_path: str,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    result: metta.tests_support.run_tool.RunToolResult = metta.tests_support.run_tool.run_tool_in_process(
        tool_path,
        "--dry-run",
        monkeypatch=monkeypatch,
        capsys=capsys,
    )

    if result.returncode != 0:
        raise RuntimeError(f"Failed to run tool {tool_path}: {result.stderr}")

    combined_output = result.stdout + result.stderr
    assert "Configuration validation successful" in combined_output
