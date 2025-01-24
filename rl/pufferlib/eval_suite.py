from typing import List, Dict
import logging
import numpy as np
from omegaconf import OmegaConf, DictConfig
from .evaluator import PufferEvaluator
from agent.policy_store import PolicyRecord

logger = logging.getLogger("eval_suite")

class EvalSuite(PufferEvaluator):
    def __init__(self, cfg: DictConfig, policy_record: PolicyRecord, baseline_records: List[PolicyRecord], **kwargs):
        super().__init__(cfg, policy_record, baseline_records)
        #get the custom metrics from the evaluator config
        self.metrics = cfg.get("custom_metrics", [])

    def compute_time_to_reach(self, game_stats: List[Dict], target: str) -> float:
        times = []
        for episode in game_stats:
            for agent in episode:
                if f"reached_{target}" in agent:
                    times.append(agent[f"reached_{target}"])
        return np.mean(times) if times else float('inf')

    def compute_metric(self, game_stats: List[Dict], metric_cfg: Dict) -> float:
        if metric_cfg["type"] == "time_to_reach":
            return self.compute_time_to_reach(game_stats, metric_cfg["target"])
        return 0.0

    def evaluate(self):
        game_stats = super().evaluate()
        
        eval_results = {
            "env": self._cfg.env,
            "policy": self._policy_name,
            "metrics": {}
        }

        #TODO: have to actually calculate the metrics here, 
        #i think i need to get trajectories out of the evaluator in order to do so
        
        for metric in self.metrics:
            value = self.compute_metric(game_stats, metric)
            eval_results["metrics"][metric["name"]] = value
            
        return eval_results