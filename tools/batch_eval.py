import logging

import hydra
from omegaconf import DictConfig
from util.runtime_configuration import setup_mettagrid_environment
from eval import simulate_policies

@hydra.main(version_base=None, config_path="../configs", config_name="eval")
def main(cfg: DictConfig):
    setup_mettagrid_environment(cfg)
    simulate_policies(cfg)
if __name__ == "__main__":
    main()
