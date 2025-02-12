import logging
from omegaconf import DictConfig, OmegaConf
from typing import Dict, Any
from rl.eval.eval_stats_db import EvalStatsDB
from tabulate import tabulate
import fnmatch
import pandas as pd
from termcolor import colored
logger = logging.getLogger("eval_stats_analyzer")

class EvalStatsAnalyzer:
    def __init__(
        self,
        stats_db: EvalStatsDB,
        analysis: DictConfig,
        **kwargs):
        self.analysis = analysis
        self.stats_db = stats_db
        self.global_filters = analysis.get('filters', None)

    def _filters(self, item):
        filters = item.get('filters', None)
        if self.global_filters:
            filters = {**self.global_filters, **filters} if filters else self.global_filters
        return filters

    @staticmethod
    def log_significance(comparison_data, metrics, filters):
        comparison_df = pd.DataFrame(comparison_data, columns=['Policy 1', 'Policy 2', 'p-value', 'effect size', 'interpretation', 'metric'])
        logger.info(f"\nSignificance test results for {metrics}:\n" + f"with filters {filters}" +
                   tabulate(comparison_df, headers='keys', tablefmt='grid'))


    def log_result(self, result, metric, filters, significance):
        result_table = tabulate(result, headers=["policy_name"] + list(result.keys()), tablefmt="grid", maxcolwidths=25)
        logger.info(f"Results for {metric} with filters {filters}:\n{result_table}")
        if len(significance) > 0:
            self.log_significance(significance, metric, filters)

    def analyze(self):
        metric_configs = {}
        for cfg in self.analysis.metrics:
            metric = cfg.metric
            if '*' in metric or '?' in metric:
                # Use fnmatch.filter to match against the available metrics
                metric_configs[cfg] = fnmatch.filter(self.stats_db.available_metrics, metric)
            else:
                metric_configs[cfg] = [metric]

        filters = None
        for cfg, metrics in metric_configs.items():
            filters = self._filters(cfg)
            group_by_episode = cfg.get('group_by_episode', False)
            result, significance = self.stats_db.analyze_policies(metrics, filters, group_by_episode)

            if len(result) == 0:
                logger.info(f"No data found for {metrics} with filters {filters}" + "\n")
                continue

            self.log_result(result, metrics, filters, significance)
