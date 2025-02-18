import logging
from omegaconf import DictConfig
from typing import Dict, Any, List, Optional
from rl.eval.eval_stats_db import EvalStatsDB
from tabulate import tabulate
import fnmatch
import pandas as pd
from rl.eval.stats import significance_test
logger = logging.getLogger("eval_stats_analyzer")

class EvalStatsAnalyzer:
    def __init__(
        self,
        stats_db: EvalStatsDB,
        analysis: DictConfig,
        policy_uri: str,
        **kwargs):
        self.analysis = analysis
        self.stats_db = stats_db
        self.candidate_policy_uri = policy_uri
        self.global_filters = analysis.get('filters', None)

    def _filters(self, item):
        local_filters = item.get('filters')

        if not self.global_filters:
            return local_filters

        if not local_filters:
            return self.global_filters

        merged = {}
        for key in set(self.global_filters) | set(local_filters):
            global_val = self.global_filters.get(key)
            local_val = local_filters.get(key)
            # If both values exist and are lists, concatenate them.
            if isinstance(global_val, list) and isinstance(local_val, list):
                merged[key] = global_val + local_val
            # Otherwise, prefer the local value if it exists; if not, use the global value.
            else:
                merged[key] = local_val if key in local_filters else global_val

        return merged

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

    def _analyze_metrics(self, metric_fields: List[str], filters: Optional[Dict[str, Any]] = None, group_by_episode: bool = False) -> pd.DataFrame:
        result_dfs = []
        significance_results = []
        for metric in metric_fields:
            if not metric in self.stats_db.available_metrics:
                logger.info(f"Metric {metric} not found in stats_db")
                continue
            df_per_episode = self.stats_db._metric(metric, filters, group_by_episode)

            df_per_episode, df_metric = self.stats_db._metric(metric, filters, group_by_episode)
            result_dfs.append(df_metric)

            # Only calculate significance if there are at least 2 policies
            if df_per_episode.shape[1] > 1:
                significance_results += significance_test(df_per_episode, metric)

        metrics_df = pd.concat(result_dfs, axis=1) if len(result_dfs) > 0 else None

        return metrics_df, significance_results

    def analyze(self):
        p_fitness = None
        metric_configs = {}
        for cfg in self.analysis.metrics:
            metric_configs[cfg] = fnmatch.filter(self.stats_db.available_metrics, cfg.metric)
        results = []
        significances = []

        filters = None
        for cfg, metrics in metric_configs.items():
            filters = self._filters(cfg)
            group_by_episode = cfg.get('group_by_episode', False)
            result, significance = self._analyze_metrics(metrics, filters, group_by_episode)

            if result is None:
                logger.info(f"No data found for {metrics} with filters {filters}" + "\n")
                continue

            self.log_result(result, metrics, filters, significance)
            results.append(result)
            significances.append(significance)

        if self.candidate_policy_uri is not None:
            uri = self.candidate_policy_uri.replace("wandb://run/", "")
            matching_indices = [i for i in results[0].index if uri in i]
            if matching_indices:
                uri = matching_indices[0]

            fitness = self.policy_fitness(results)
            results.append(fitness)

        return results, significances

    def policy_fitness(self, metric_data):
        if self.analysis.baseline is None:
            return metric_data.loc[self.candidate_policy_uri]
            average_metric = metric_data.mean(axis=1)
