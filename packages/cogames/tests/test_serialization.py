from pathlib import Path

import torch
from typer.testing import CliRunner

from cogames.cogs_vs_clips.scenarios import make_game
from cogames.examples.simple_policy import SimplePolicy
from cogames.main import app
from cogames.serialization import bundle_policy, load_policy_from_bundle, save_policy
from mettagrid import MettaGridEnv

runner = CliRunner()


def _make_env():
    cfg = make_game(num_cogs=1)
    return MettaGridEnv(env_cfg=cfg)


def test_bundle_and_load(tmp_path: Path) -> None:
    env = _make_env()
    policy = SimplePolicy(env, torch.device("cpu"))
    artifact = save_policy(policy, tmp_path / "raw")
    bundle_dir = tmp_path / "bundle"
    bundle_policy(artifact.policy_class, artifact.weights_path, bundle_dir)

    env2 = _make_env()
    loaded = load_policy_from_bundle(bundle_dir, env2, torch.device("cpu"))

    assert isinstance(loaded, SimplePolicy)


def test_policy_export_cli(tmp_path: Path) -> None:
    env = _make_env()
    policy = SimplePolicy(env, torch.device("cpu"))
    artifact = save_policy(policy, tmp_path / "raw")
    bundle_dir = tmp_path / "bundle"

    result = runner.invoke(
        app,
        [
            "policy",
            "export",
            artifact.policy_class,
            str(artifact.weights_path),
            str(bundle_dir),
        ],
    )

    if result.exception:
        raise result.exception

    assert result.exit_code == 0
    assert (bundle_dir / "policy.json").exists()
    assert (bundle_dir / "policy.pt").exists()
