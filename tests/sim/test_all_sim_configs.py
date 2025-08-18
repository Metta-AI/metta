import pytest
from hydra import compose, initialize_config_dir
from hydra.core.global_hydra import GlobalHydra
from omegaconf import OmegaConf

from metta.common.util.fs import get_repo_root
from metta.common.util.resolvers import register_resolvers


@pytest.fixture(scope="session", autouse=True)
def setup_resolvers():
    """Register custom OmegaConf resolvers once for the entire test session."""
    register_resolvers()


def get_all_sim_configs() -> list[str]:
    config_dir = get_repo_root() / "configs"
    sim_config_dir = config_dir / "sim"

    sim_configs = []
    for f in sim_config_dir.glob("*.yaml"):
        if f.name not in [
            "sim_suite.yaml",  # Base class for simulation suites, not a concrete config
            "sim.yaml",  # Default values for individual simulations, not a suite
            "defaults.yaml",  # Hydra defaults file, not a simulation config
            "sim_single.yaml",  # Special config for running single environments, not a suite
        ]:
            sim_configs.append(f.stem)

    return sorted(sim_configs)


@pytest.mark.slow
@pytest.mark.parametrize("sim_config", get_all_sim_configs())
def test_all_sim_configs_valid(sim_config: str):
    config_dir = get_repo_root() / "configs"

    GlobalHydra.instance().clear()

    try:
        with initialize_config_dir(config_dir=str(config_dir.resolve()), version_base=None):
            cfg = compose(config_name=f"sim/{sim_config}")

            # The config has 'sim' key which contains the simulations
            if "sim" not in cfg:
                pytest.fail(f"Sim config {sim_config} missing 'sim' key. Found keys: {list(cfg.keys())}")

            if not hasattr(cfg.sim, "simulations"):
                pytest.fail(f"Sim config {sim_config} has no simulations defined")

            if cfg.sim.simulations is None:
                pytest.fail(f"Sim config {sim_config} has empty simulations")

            for sim_name, sim_cfg in cfg.sim.simulations.items():
                if "env" not in sim_cfg:
                    pytest.fail(f"Simulation '{sim_name}' in {sim_config} missing 'env' key")

                env_path = sim_cfg.env

                GlobalHydra.instance().clear()

                with initialize_config_dir(config_dir=str(config_dir.resolve()), version_base=None):
                    try:
                        env_cfg = compose(config_name=env_path)
                        OmegaConf.resolve(env_cfg)
                    except Exception as e:
                        pytest.fail(
                            f"Failed to load env config '{env_path}' referenced by '{sim_name}' in {sim_config}: {e}"
                        )

    except Exception as e:
        pytest.fail(f"Failed to load sim config {sim_config}: {e}")
