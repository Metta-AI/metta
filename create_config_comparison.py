#!/usr/bin/env -S uv run --script
"""Create a configuration comparison between Metta and 3rd party library."""

import sys

import hydra
from omegaconf import DictConfig, OmegaConf


@hydra.main(version_base=None, config_path="configs", config_name="train_job")
def create_comparison(cfg: DictConfig) -> None:
    """Create a configuration comparison file."""

    # Resolve all configurations
    _ = OmegaConf.to_container(cfg, resolve=True)

    with open("config_comparison.txt", "w") as f:
        f.write("# Configuration Comparison: Metta vs 3rd Party Library\n\n")
        f.write("## Command Executed\n")
        f.write("Metta: ./tools/train.py run=relh.dummy.run\n")
        f.write("3rd Party: (unknown command)\n\n")

        f.write("## Key Differences and Mappings\n\n")

        # Basic Settings
        f.write("### Basic Settings\n")
        f.write("| Parameter | Metta | 3rd Party |\n")
        f.write("|-----------|--------|-----------|")
        f.write(f"\n| name | {cfg.get('run', 'unknown')} | GDY-MettaGrid |")
        f.write("\n| report_stats_interval | N/A | 100 |")
        f.write("\n| normalize_rewards | N/A | false |")
        f.write("\n| sampling | 0 (from mettagrid.yaml) | 0 |")
        f.write("\n| desync_episodes | N/A | true |\n\n")

        # Game Configuration
        f.write("### Game Configuration\n")
        f.write("| Parameter | Metta | 3rd Party |\n")
        f.write("|-----------|--------|-----------|")

        # Get the environment config
        env_path = cfg.trainer.get("curriculum", cfg.trainer.get("env", ""))
        env_name = env_path.split("/")[-1] if "/" in env_path else env_path

        # Assuming arena/basic config based on the path
        f.write(f"\n| num_agents | 64 ({env_name}) | 64 |")
        f.write("\n| obs_width | 11 | 11 |")
        f.write("\n| obs_height | 11 | 11 |")
        f.write("\n| num_observation_tokens | 200 | 200 |")
        f.write("\n| max_steps | 1000 | 1000 |\n\n")

        # Agent Configuration
        f.write("### Agent Configuration\n")
        f.write("| Parameter | Metta | 3rd Party |\n")
        f.write("|-----------|--------|-----------|")
        f.write("\n| default_resource_limit | 10 | 10 |")
        f.write("\n| resource_limits.heart | 255 | 255 |")
        f.write("\n| freeze_duration | 10 | 10 |")
        f.write("\n| action_failure_penalty | 0.0 | 0.0 |\n\n")

        # Training Parameters
        f.write("### Training Parameters\n")
        f.write("| Parameter | Metta | 3rd Party |\n")
        f.write("|-----------|--------|-----------|")

        trainer = cfg.trainer
        f.write(f"\n| total_timesteps | {trainer.get('total_timesteps', 'N/A'):,} | 300,000,000 |")
        f.write(f"\n| batch_size | {trainer.get('batch_size', 'N/A'):,} | auto |")
        f.write(f"\n| minibatch_size | {trainer.get('minibatch_size', 'N/A'):,} | 32,768 |")
        f.write(f"\n| bptt_horizon | {trainer.get('bptt_horizon', 'N/A')} | 64 |")

        # Optimizer settings
        optimizer = trainer.get("optimizer", {})
        f.write(f"\n| learning_rate | {optimizer.get('learning_rate', 'N/A')} | 0.018470110879570414 |")
        f.write(f"\n| adam_beta1 | {optimizer.get('beta1', 'N/A')} | 0.8923106632311335 |")
        f.write(f"\n| adam_beta2 | {optimizer.get('beta2', 'N/A')} | 0.9632470625784862 |")
        f.write(f"\n| adam_eps | {optimizer.get('eps', 'N/A')} | 1.3537431449843922e-7 |")

        # PPO settings
        ppo = trainer.get("ppo", {})
        f.write(f"\n| clip_coef | {ppo.get('clip_coef', 'N/A')} | 0.14919147162017737 |")
        f.write(f"\n| ent_coef | {ppo.get('ent_coef', 'N/A')} | 0.016700174334611493 |")
        f.write(f"\n| gae_lambda | {ppo.get('gae_lambda', 'N/A')} | 0.8443676864928215 |")
        f.write(f"\n| gamma | {ppo.get('gamma', 'N/A')} | 0.997950174315581 |")
        f.write(f"\n| max_grad_norm | {ppo.get('max_grad_norm', 'N/A')} | 2.572849891206465 |")
        f.write(f"\n| vf_clip_coef | {ppo.get('vf_clip_coef', 'N/A')} | 0.1569624916309049 |")
        f.write(f"\n| vf_coef | {ppo.get('vf_coef', 'N/A')} | 3.2211333828684454 |")

        # VTrace settings
        vtrace = trainer.get("vtrace", {})
        f.write(f"\n| vtrace_c_clip | {vtrace.get('vtrace_c_clip', 'N/A')} | 2.134490283650365 |")
        f.write(f"\n| vtrace_rho_clip | {vtrace.get('vtrace_rho_clip', 'N/A')} | 2.296343917695581 |")

        # Prioritized experience replay
        per = trainer.get("prioritized_experience_replay", {})
        f.write(f"\n| prio_alpha | {per.get('prio_alpha', 'N/A')} | 0.7918451491719373 |")
        f.write(f"\n| prio_beta0 | {per.get('prio_beta0', 'N/A')} | 0.5852686803034238 |\n\n")

        # Environment/Map Configuration
        f.write("### Environment/Map Configuration\n")
        f.write("| Parameter | Metta | 3rd Party |\n")
        f.write("|-----------|--------|-----------|")
        f.write("\n| map width | 64 | 64 |")
        f.write("\n| map height | 64 | 64 |")
        f.write("\n| border_width | 6 | 6 |")
        f.write("\n| num_rooms | N/A | 1 |\n\n")

        # Object Counts
        f.write("### Object Counts (arena/basic vs 3rd party)\n")
        f.write("| Object | Metta | 3rd Party |\n")
        f.write("|--------|--------|-----------|")
        f.write("\n| mine_red | 20 | 128 |")
        f.write("\n| generator_red | 10 | 64 |")
        f.write("\n| altar | 8 | 32 |")
        f.write("\n| wall | 40 | 0 |")
        f.write("\n| block | 40 | N/A |")
        f.write("\n| armory | N/A | 0 |")
        f.write("\n| lasery | N/A | 0 |")
        f.write("\n| lab | N/A | 0 |")
        f.write("\n| factory | N/A | 0 |")
        f.write("\n| temple | N/A | 0 |\n\n")

        # Reward Settings
        f.write("### Reward Settings\n")
        f.write("| Parameter | Metta | 3rd Party |\n")
        f.write("|-----------|--------|-----------|")
        f.write("\n| ore_reward | N/A | 0.17088483842567775 |")
        f.write("\n| battery_reward | N/A | 0.9882859711234822 |")
        f.write("\n| heart_reward | 1 (from agent rewards) | 1.0 |\n\n")

        # Key Structural Differences
        f.write("### Key Structural Differences\n\n")
        f.write("1. **Configuration System**: \n")
        f.write("   - Metta uses Hydra with compositional configs\n")
        f.write("   - 3rd party uses a single flat configuration file\n\n")

        f.write("2. **Agent Architecture**:\n")
        f.write("   - Metta uses a modular component system with CNN + LSTM\n")
        f.write("   - 3rd party architecture not specified in config\n\n")

        f.write("3. **Optimization**:\n")
        f.write("   - Metta uses more conservative hyperparameters (lower LR, lower entropy)\n")
        f.write("   - 3rd party uses more aggressive settings with higher learning rate\n\n")

        f.write("4. **Map Generation**:\n")
        f.write("   - Metta uses MapGen with instances\n")
        f.write("   - 3rd party uses MultiRoom with single room\n\n")

        f.write("5. **Training Duration**:\n")
        f.write("   - Metta configured for 10B steps (33x longer)\n")
        f.write("   - 3rd party configured for 300M steps\n\n")

        f.write("6. **Batch Sizes**:\n")
        f.write("   - Metta: larger batch size (524k vs auto)\n")
        f.write("   - 3rd party: larger minibatch size (32k vs 16k)\n\n")

        f.write("7. **Prioritized Experience Replay**:\n")
        f.write("   - Metta: disabled (prio_alpha=0.0)\n")
        f.write("   - 3rd party: enabled (prio_alpha=0.79)\n\n")

        f.write("8. **V-trace Clipping**:\n")
        f.write("   - Metta: conservative (1.0)\n")
        f.write("   - 3rd party: more permissive (2.13, 2.29)\n\n")

        f.write("### Notable Missing Configurations in Metta's Resolved Config\n\n")
        f.write("1. Explicit reward values for ore and battery\n")
        f.write("2. Recipe details for converters\n")
        f.write("3. Cooldown values for objects\n")
        f.write("4. Conversion tick rates\n")
        f.write("5. Object type IDs\n")
        f.write("6. Input/output resource specifications for converters\n\n")

        f.write("### Recommendations for Performance Comparison\n\n")
        f.write("1. The 3rd party config has much more aggressive hyperparameters which could lead to faster\n")
        f.write("2. The prioritized experience replay is a significant difference - Metta has it disabled\n")
        f.write("3. The map sizes are now aligned (64x64) which improves comparability\n")
        f.write("4. The object densities are still different - 3rd party has more objects per map\n")
        f.write("5. Consider aligning batch/minibatch sizes for fair comparison\n")
        f.write("6. The learning rate difference is substantial (40x higher in 3rd party)\n")

    print("Updated config comparison written to config_comparison.txt")


if __name__ == "__main__":
    # Override sys.argv to pass the correct arguments
    sys.argv = ["create_config_comparison.py", "run=relh.dummy.run"]
    create_comparison()
