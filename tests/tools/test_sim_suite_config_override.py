import json

from omegaconf import OmegaConf

from metta.sim.simulation_config import SimulationSuiteConfig, SingleEnvSimulationConfig


def test_sim_suite_config_cli_roundtrip():
    # Build a sample SimulationSuiteConfig as dict
    sim_cfg = SimulationSuiteConfig(
        name="arena",
        simulations={
            "eval/env1": SingleEnvSimulationConfig(
                env="eval/env1", num_episodes=3, env_overrides={"game": {"max_steps": 100}}
            ),
            "eval/env2": SingleEnvSimulationConfig(env="eval/env2", num_episodes=2),
        },
        env_overrides={"game": {"num_agents": 4}},
        num_episodes=10,
    )

    # Serialize to JSON string (as the worker does)
    json_str = json.dumps(sim_cfg.model_dump(mode="json"))

    # Simulate Hydra CLI injection: sim_suite_config=<json>
    cfg = OmegaConf.from_cli([f"sim_suite_config={json_str}"])

    # Ensure CLI parsed into structured data, not a plain string
    assert not isinstance(cfg.sim_suite_config, str)
    sim_suite_dict = OmegaConf.to_container(cfg.sim_suite_config, resolve=True)  # type: ignore[arg-type]
    assert isinstance(sim_suite_dict, dict)
    assert sim_suite_dict["name"] == "arena"

    # Deserialize into strongly-typed SimulationSuiteConfig (as sim.py does)
    parsed = SimulationSuiteConfig.model_validate(sim_suite_dict)

    # Validate key fields survived the roundtrip
    assert parsed.name == sim_cfg.name
    assert parsed.num_episodes == sim_cfg.num_episodes
    assert parsed.env_overrides == sim_cfg.env_overrides

    # Validate per-env config
    assert "eval/env1" in parsed.simulations
    assert parsed.simulations["eval/env1"].env == "eval/env1"
    assert parsed.simulations["eval/env1"].num_episodes == 3
    assert parsed.simulations["eval/env1"].env_overrides == {"game": {"max_steps": 100}}
