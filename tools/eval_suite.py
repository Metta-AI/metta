import logging
import os
import signal
import hydra
from omegaconf import OmegaConf
from rich import traceback
from rl.wandb.wandb_context import WandbContext
from util.seeding import seed_everything
from agent.policy_store import PolicyStore
from rl.pufferlib.eval_suite import EvalSuite

logger = logging.getLogger("eval_suite.py")
signal.signal(signal.SIGINT, lambda sig, frame: os._exit(0))

@hydra.main(version_base=None, config_path="../configs", config_name="config")
def main(cfg):
    traceback.install(show_locals=False)
    logger.info(OmegaConf.to_yaml(cfg))
    seed_everything(cfg.seed, cfg.torch_deterministic)

    with WandbContext(cfg) as wandb_run:
        policy_store = PolicyStore(cfg, wandb_run)
        eval_suite_cfg = cfg.eval_suite
        
        results = []
        for eval_cfg in eval_suite_cfg.evaluators:
            env_cfg = OmegaConf.load(f"configs/eval_suite/evals/{eval_cfg.env}.yaml")
            #evaluator_cfg = OmegaConf.merge(cfg.evaluator, eval_cfg.evaluator)
            eval_cfg.env = env_cfg

            merged_cfg = OmegaConf.merge(cfg, eval_cfg)
            
            policy = policy_store.policy(merged_cfg.evaluator.policy)
            baselines = policy_store.policies(merged_cfg.evaluator.baselines) if merged_cfg.evaluator.baselines.uri else []

            #merged_cfg.env.num_agents = env_cfg.game.num_agents

            evaluator = hydra.utils.instantiate(merged_cfg.evaluator, merged_cfg, policy, baselines)
            
            evaluator = EvalSuite(merged_cfg, policy, baselines) #Evaluator -> runs evaluation and calculaes stats


            result = evaluator.evaluate()
            evaluator.close()
            results.append(result)
            
            logger.info(f"\nResults for {result['env']} using {result['policy']}:")
            for metric_name, value in result['metrics'].items():
                logger.info(f"{metric_name}: {value}")
                if wandb_run:
                    wandb_run.log({f"{result['env']}/{metric_name}": value})
                    
        #log stats for the entire suite

if __name__ == "__main__":
    main()