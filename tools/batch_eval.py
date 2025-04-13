
import hydra
from omegaconf import DictConfig

from eval import simulate_policies
from util.runtime_configuration import setup_mettagrid_environment


@hydra.main(version_base=None, config_path="../configs", config_name="eval")
def main(cfg: DictConfig):
    setup_mettagrid_environment(cfg)
    simulate_policies(cfg)
if __name__ == "__main__":
    main()
