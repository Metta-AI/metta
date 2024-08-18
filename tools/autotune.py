from omegaconf import OmegaConf
import pufferlib.vector
import hydra

def make_env():
    global env_config
    env = hydra.utils.instantiate(env_config.env, render_mode="human")
    env.emulated = None
    env.single_observation_space = env.observation_space
    env.single_action_space = env.action_space
    env.num_agents = env.player_count
    env.done = False
    return env

@hydra.main(version_base=None, config_path="../configs", config_name="config")
def main(cfg):
    global env_config
    env_config = cfg
    pufferlib.vector.autotune(make_env, batch_size=16320//20, max_envs=1024, max_env_ram_gb=64)
    # pufferlib.vector.autotune(make_env, batch_size=16384//20)

if __name__ == "__main__":
    main()
