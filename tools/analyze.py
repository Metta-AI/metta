import hydra
from omegaconf import DictConfig
from util.runtime_configuration import setup_metta_environment
from eval.report import generate_report
import logging
@hydra.main(version_base=None, config_path="../configs", config_name="analyzer")
def main(cfg: DictConfig) -> None:
    setup_metta_environment(cfg)
    logger = logging.getLogger(__name__)
    view_type = "latest"
    logger.info(f"Generating {view_type} report")
    generate_report(cfg)


if __name__ == "__main__":
    main()