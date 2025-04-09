import hydra
from omegaconf import DictConfig
from util.runtime_configuration import setup_metta_environment
from eval.report import generate_report

@hydra.main(version_base=None, config_path="../configs", config_name="analyzer")
def main(cfg: DictConfig) -> None:
    setup_metta_environment(cfg)
    generate_report(cfg)

if __name__ == "__main__":
    main()