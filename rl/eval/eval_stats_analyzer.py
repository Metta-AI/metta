import logging
from omegaconf import DictConfig
from typing import Dict, Any, List, Optional
from rl.eval.eval_stats_db import EvalStatsDB
from tabulate import tabulate
import fnmatch
import numpy as np
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
        result_table = tabulate(result, headers=["policy_and_eval"] + list(result.keys()), tablefmt="grid", maxcolwidths=25)
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
            df_per_episode, df_metric = self.stats_db._metric(metric, filters, group_by_episode)
            result_dfs.append(df_metric)

            # Only calculate significance if there are at least 2 policies
            if df_per_episode.shape[1] > 1:
                significance_results += significance_test(df_per_episode, metric)

        metrics_df = pd.concat(result_dfs, axis=1) if len(result_dfs) > 0 else None

        return metrics_df, significance_results

    def analyze(self):
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
            policy_fitness = self.policy_fitness(results)

        if self.candidate_policy_uri is not None:
            for metric, fitness in policy_fitness.items():
                wandb.log({
                    f"policy_fitness/{metric.name}": fitness[0],
                    f"policy_fitness/{metric.name}_std": fitness[1]
                })

        return results, significances, policy_fitness

    @staticmethod
    def get_latest_policy(all_policies, uri):
        if uri in all_policies:
            return uri
        policy_versions = [i for i in all_policies if uri in i]
        policy_versions.sort(key=lambda x: int(x.split(':v')[-1]))
        candidate_uri = policy_versions[-1]
        return candidate_uri

    def policy_fitness(self, metric_data):
        policy_fitness = {}
        uri = self.candidate_policy_uri.replace("wandb://run/", "")

        all_policies = metric_data[0].index

        # Get the latest version of the candidate policy
        candidate_uri = self.get_latest_policy(all_policies, uri)

        baseline_policies = list(set([self.get_latest_policy(all_policies, b) for b in self.analysis.baseline_policies or all_policies]))

        for metric in metric_data:


            candidate_metric_mean, candidate_metric_std = metric.loc[candidate_uri]
            baseline_metric_mean, baseline_metric_std = np.mean([metric.loc[baseline_uri] for baseline_uri in baseline_policies],axis=0)

            fitness = (candidate_metric_mean - baseline_metric_mean) / baseline_metric_std
            policy_fitness[metric] = fitness
        return policy_fitness