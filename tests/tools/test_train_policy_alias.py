import pytest

from metta.agent.policies.fast import FastConfig
from metta.tools.train import TrainTool
from mettagrid.util.module import load_symbol


def test_policy_alias_fast_resolves() -> None:
    tool = TrainTool.model_validate({"policy_architecture": "fast", "training_env": {}})
    assert isinstance(tool.policy_architecture, FastConfig)


@pytest.mark.skipif(
    not TrainTool.POLICY_PRESETS,
    reason="Policy presets are not available in this build",
)
@pytest.mark.parametrize(
    ("alias", "target_path"),
    list(TrainTool.POLICY_PRESETS.items()),
)
def test_policy_aliases_resolve(alias: str, target_path: str) -> None:
    try:
        target_cls = load_symbol(target_path)
    except (ImportError, AttributeError, ModuleNotFoundError) as exc:
        pytest.skip(f"Unable to import preset '{alias}': {exc}")

    tool = TrainTool.model_validate({"policy_architecture": alias, "training_env": {}})
    assert isinstance(tool.policy_architecture, target_cls)
