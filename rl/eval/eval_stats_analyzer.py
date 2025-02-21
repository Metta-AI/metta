import logging
from omegaconf import DictConfig
from typing import Dict, Any, List, Optional
from rl.eval.eval_stats_db import EvalStatsDB
from tabulate import tabulate
import fnmatch
import numpy as np
import pandas as pd
import os
import wandb
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

    def log_result(self, result, metric, filters):
        result_table = tabulate(result, headers=list(result.keys()), tablefmt="grid", maxcolwidths=25)
        logger.info(f"Results for {metric} with filters {filters}:\n{result_table}")


    def _analyze_metrics(self, metric_configs):
        result_dfs = []
        policy_fitness_records = []
        for cfg, metrics in metric_configs.items():
            filters = self._filters(cfg)
            group_by_episode = cfg.get('group_by_episode', False)
            for metric in metrics:
                metric_result = self.stats_db._metric(metric, filters, group_by_episode)
                if len(metric_result) == 0:
                    logger.info(f"No data found for {metric} with filters {filters}" + "\n")
                    continue
                policy_fitness = self.policy_fitness(metric_result, metric)
                policy_fitness_records.extend(policy_fitness)
                result_dfs.append(metric_result)
                # self.log_result(metric_result, metric, filters)

        return result_dfs, policy_fitness_records

    def analyze(self):
        metric_configs = {}
        for cfg in self.analysis.metrics:
            metric_configs[cfg] = fnmatch.filter(self.stats_db.available_metrics, cfg.metric)
        if all(len(metric_configs[cfg]) == 0 for cfg in metric_configs):
            logger.info(f"No metrics to analyze yet for {self.candidate_policy_uri}")
            return [], []
        result_dfs = []
        policy_fitness_records = []

        result_dfs, policy_fitness_records = self._analyze_metrics(metric_configs)

        policy_fitness_df = pd.DataFrame(policy_fitness_records)
        if len(policy_fitness_df) > 0:
            policy_fitness_table = tabulate(policy_fitness_df, headers=[self.candidate_policy_uri] + list(policy_fitness_df.keys()), tablefmt="grid", maxcolwidths=25)
            logger.info(f"Policy fitness results for candidate policy {self.candidate_policy_uri} and baselines {self.analysis.baseline_policies}:\n{policy_fitness_table}")

        return result_dfs, policy_fitness_records

    @staticmethod
    def get_latest_policy(all_policies, uri):
        if uri in all_policies:
            return uri
        policy_versions = [i for i in all_policies if uri in i]
        if len(policy_versions) == 0:
            raise ValueError(f"No policy found in DB for candidate policy: {uri}, options are {all_policies}")
        if len(policy_versions) > 1 and 'wandb' in uri:
            policy_versions.sort(key=lambda x: int(x.split(':v')[-1]))
        candidate_uri = policy_versions[-1]

        return candidate_uri

    def policy_fitness(self, metric_data, metric_name):
        policy_fitness = []
        if "wandb" in self.candidate_policy_uri:
            uri = self.candidate_policy_uri.replace("wandb://run/", "")
        elif "file" in self.candidate_policy_uri:
            uri = self.candidate_policy_uri.replace("file://", "")
        else:
            uri = self.candidate_policy_uri

        all_policies = metric_data['policy_name'].unique()

        # Get the latest version of the candidate policy
        candidate_uri = self.get_latest_policy(all_policies, uri)

        baseline_policies = list(set([self.get_latest_policy(all_policies, b) for b in self.analysis.baseline_policies or all_policies]))

        metric_data = metric_data.set_index('policy_name')
        eval, metric_mean, metric_std = metric_data.keys()

        candidate_data = metric_data.loc[candidate_uri].set_index(eval)
        baseline_data = metric_data.loc[baseline_policies].set_index(eval)

        evals = metric_data[eval].unique()

        for eval in evals:
            # Is the difference the correct way to do this?
            candidate_mean = candidate_data.loc[eval][metric_mean]
            baseline_mean = np.mean(baseline_data.loc[eval][metric_mean])
            fitness = (candidate_data.loc[eval][metric_mean] - np.mean(baseline_data.loc[eval][metric_mean])) # / np.std(baseline_data.loc[eval][metric_std])
            policy_fitness.append({"eval": eval, "metric": metric_name, "candidate_mean": candidate_mean, "baseline_mean": baseline_mean, "fitness": fitness})
        return policy_fitness
