# suite/eval_suite.py

import logging
import copy
import hydra
from omegaconf import DictConfig, OmegaConf

# Import your evaluator class
from rl.pufferlib.evaluator import PufferEvaluator

logger = logging.getLogger(__name__)


@hydra.main(version_base=None, config_path=".", config_name="suite_config")
def main(cfg: DictConfig):
    """
    This function loads suite_config.yaml, which has:
      policy_uri: ...
      baseline_uri: ...
      evals:
        - { env: multi_room_cylinder, metrics_to_log: [...] }
        - ...
    Then for each item in `evals`, we merge the corresponding environment config
    and run evaluation.
    """
    logger.info("===== Beginning Evaluation Suite =====")

    # The top-level policy/baseline URIs from suite_config.yaml
    logger.info(f"Global Policy URI: {cfg.policy_uri}")
    logger.info(f"Global Baseline URI: {cfg.baseline_uri}")

    # We iterate over each evaluation item in the list
    for eval_item in cfg.evals:
        env_name = eval_item.env  # e.g. "multi_room_cylinder"
        metrics_to_log = eval_item.get("metrics_to_log", [])
        npcs = eval_item.get("npcs", 0)

        # 1) Create a working copy of the base config
        #    (which currently includes the base evaluator config).
        eval_cfg = copy.deepcopy(cfg)

        # 2) Merge environment config into our copy
        #    We load the environment override from /envs/<env_name>.yaml
        env_override = hydra.compose(config_name=f"{env_name}.yaml", 
                                     overrides=[f"env.game.num_agents={4 + npcs}"])  
        # ^ Example of dynamically injecting "npcs" into `env.game.num_agents`.
        #   Adjust as needed to reflect your multi-agent logic.

        # Now we merge env_override into eval_cfg:
        eval_cfg = OmegaConf.merge(eval_cfg, env_override)

        # 3) Insert the policy/baseline URIs into the correct place in eval_cfg
        #    (Assuming your base evaluator config stores these in `eval_cfg.policy.uri` etc.)
        eval_cfg.policy.uri = cfg.policy_uri
        eval_cfg.baselines.uri = cfg.baseline_uri

        logger.info(f"--- Running evaluation for environment: {env_name} ---")
        logger.info(f"Metrics to log: {metrics_to_log}")
        logger.info(f"Merged config:\n{OmegaConf.to_yaml(eval_cfg, resolve=True)}")

        # 4) Instantiate the evaluator
        #    (If PufferEvaluator only needs (eval_cfg, policy_record, baseline_records),
        #     you'll need to create the policy/baseline records. Here's a simplified placeholder.)
        dummy_policy_record = ...  # e.g. your PolicyRecord from policy_store
        dummy_baselines = [...]    # e.g. list of baseline PolicyRecords
        evaluator = PufferEvaluator(eval_cfg, dummy_policy_record, dummy_baselines)

        # 5) Run evaluate()
        stats = evaluator.evaluate()

        # 6) Possibly log `metrics_to_log` here or pass them somewhere
        #    This is entirely up to how you want to handle additional custom metrics.
        #    For example, if "metrics_to_log" refers to your categories in eval_cfg.stat_categories,
        #    you might run some analysis on `stats` here.

        evaluator.close()

    logger.info("===== All Evaluations Complete =====")


if __name__ == "__main__":
    main()
