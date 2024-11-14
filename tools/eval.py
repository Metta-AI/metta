import os
import signal  # Aggressively exit on ctrl+c
import json # For loading historical elo or glicko scores from local disk
import hydra
from omegaconf import OmegaConf
from rich import traceback
from rl.wandb.wandb_context import WandbContext
from util.seeding import seed_everything
from agent.policy_store import PolicyStore
from util.stats_library import Glicko2Test, MannWhitneyUTest, EloTest, StatisticalTest

signal.signal(signal.SIGINT, lambda sig, frame: os._exit(0))

@hydra.main(version_base=None, config_path="../configs", config_name="config")
def main(cfg):
    traceback.install(show_locals=False)
    print(OmegaConf.to_yaml(cfg))
    seed_everything(cfg.seed, cfg.torch_deterministic)

    with WandbContext(cfg) as wandb_run:
        policy_store = PolicyStore(cfg, wandb_run)

        evaluator = hydra.utils.instantiate(cfg.evaluator, cfg, policy_store)
        stats = evaluator.evaluate()
        evaluator.close()

        print_test_results(MannWhitneyUTest(stats, cfg.evaluator.stat_categories['all']))
        print_test_results(EloTest(stats, cfg.evaluator.stat_categories['altar']), cfg.evaluator.baselines.elo_scores_path)
        print_test_results(Glicko2Test(stats, cfg.evaluator.stat_categories['altar']), cfg.evaluator.baselines.glicko_scores_path)

def print_test_results(test: StatisticalTest, scores_path: str = None):
    historical_data = {}
    if scores_path and os.path.exists(scores_path):
        with open(scores_path, "r") as file:
            print(f"Loading historical data from {scores_path}")
            try:
                historical_data = json.load(file)
            except json.JSONDecodeError:
                print(f"Failed to load historical data from {scores_path}")
            test.withHistoricalData(historical_data)

    results = test.evaluate()
    print(test.format_results(results))

    if scores_path:
        os.makedirs(os.path.dirname(scores_path), exist_ok=True)
        with open(scores_path, "w") as file:
            print(f"Saving updated historical data to {scores_path}")
            historical_data.update(results)
            json.dump(historical_data, file, indent=4)

if __name__ == "__main__":
    main()
