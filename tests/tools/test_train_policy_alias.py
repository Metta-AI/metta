import importlib.util
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


def load_local_module(name: str, relative_path: str):
    module_path = ROOT / relative_path
    spec = importlib.util.spec_from_file_location(name, module_path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Unable to load module {name} from {module_path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    sys.modules[name] = module
    return module


metta_agent_policy_module = load_local_module("metta.agent.policy", "agent/src/metta/agent/policy.py")
metta_rl_training_module = load_local_module("metta.rl.training", "metta/rl/training/__init__.py")
metta_tools_train_module = load_local_module("metta.tools.train", "metta/tools/train.py")
mettagrid_util_module = load_local_module(
    "mettagrid.util.module", "packages/mettagrid/python/src/mettagrid/util/module.py"
)

# ruff: noqa: E402
FastConfig = load_local_module("metta.agent.policies.fast", "agent/src/metta/agent/policies/fast.py").FastConfig
TrainingEnvironmentConfig = load_local_module(
    "metta.rl.training.training_environment",
    "metta/rl/training/training_environment.py",
).TrainingEnvironmentConfig
TrainTool = metta_tools_train_module.TrainTool
load_symbol = mettagrid_util_module.load_symbol


def minimal_training_env() -> dict:
    return TrainingEnvironmentConfig().model_dump()


def test_policy_alias_fast_resolves() -> None:
    tool = TrainTool.model_validate({"policy_architecture": "fast", "training_env": minimal_training_env()})
    assert isinstance(tool.policy_architecture, FastConfig)


@pytest.mark.skipif(
    not TrainTool.policy_presets(),
    reason="Policy presets are not available in this build",
)
@pytest.mark.parametrize(
    ("alias", "target_path"),
    list(TrainTool.policy_presets().items()),
)
def test_policy_aliases_resolve(alias: str, target_path: str) -> None:
    try:
        target_cls = load_symbol(target_path)
    except (ImportError, AttributeError, ModuleNotFoundError) as exc:
        pytest.skip(f"Unable to import preset '{alias}': {exc}")

    tool = TrainTool.model_validate({"policy_architecture": alias, "training_env": minimal_training_env()})
    assert isinstance(tool.policy_architecture, target_cls)
