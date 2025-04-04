import logging

import hydra
from omegaconf import DictConfig, OmegaConf
from util.runtime_configuration import setup_metta_environment
from sim import simulate_policy

@hydra.main(version_base=None, config_path="../configs", config_name="eval")
def main(cfg: DictConfig):
    setup_metta_environment(cfg)
    simulate_policy(cfg)

if __name__ == "__main__":
    main()
