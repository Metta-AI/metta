import fnmatch
import logging
from typing import List, Tuple

import numpy as np
import pandas as pd
from omegaconf import ListConfig
from tabulate import tabulate

from metta.eval.analysis_config import AnalysisConfig, Metric
from metta.sim.eval_stats_db import EvalStatsDB
from metta.sim.queries import total_metric


class EvalStatsAnalyzer:
    def __init__(self, stats_db: EvalStatsDB, analysis: AnalysisConfig, policy_uri: str):
        self.logger = logging.getLogger(__name__)
        self.analysis = analysis
        self.stats_db = stats_db
        self.candidate_policy_uri = policy_uri
        self.global_filters = analysis.filters

        metric_configs: List[Tuple[Metric, List[str]]] = []
        for cfg in self.analysis.metrics:
            cur_metrics = fnmatch.filter(self.stats_db.available_metrics, cfg.metric)
            metric_configs.append((cfg, cur_metrics))
        self.metric_configs = metric_configs

    def _filters(self, item: Metric):
        local_filters = item.filters

        if len(self.global_filters) == 0:
            return local_filters

        if len(local_filters) == 0:
            return self.global_filters

        merged = {}
        for key in set(self.global_filters) | set(local_filters):
            global_val = self.global_filters.get(key)
            local_val = local_filters.get(key)
            # If both values exist and are lists, concatenate them.
            if type(global_val) in [ListConfig, list] and type(local_val) in [ListConfig, list]:
                merged[key] = global_val + local_val
            # Otherwise, prefer the local value if it exists; if not, use the global value.
            else:
                merged[key] = local_val if key in local_filters else global_val

        return merged

    def log_result(self, result, metric, filters):
        result_table = tabulate(result, headers=list(result.keys()), tablefmt="grid", maxcolwidths=25)
        self.logger.info(f"Results for {metric} with filters {filters}:\n{result_table}")

    def _analyze_metrics(self, metric_configs: List[Tuple[Metric, List[str]]], include_policy_fitness=True):
        result_dfs = []
        policy_fitness_records = []
        for cfg, metrics in metric_configs:
            filters = self._filters(cfg)
            for metric in metrics:
                metric_result = self.stats_db.query(total_metric(metric, filters))
                if len(metric_result) == 0:
                    self.logger.info(f"No data found for {metric} with filters {filters}" + "\n")
                    continue
                if include_policy_fitness:
                    policy_fitness = self.policy_fitness(metric_result, metric)
                    policy_fitness_records.extend(policy_fitness)
                result_dfs.append(metric_result)

        return result_dfs, policy_fitness_records

    def analyze(self, include_policy_fitness=True):
        if all(len(m) == 0 for (cfg, m) in self.metric_configs):
            self.logger.info(f"No metrics to analyze yet for {self.candidate_policy_uri}")
            return [], []

        result_dfs, policy_fitness_records = self._analyze_metrics(self.metric_configs, include_policy_fitness)

        if include_policy_fitness:
            policy_fitness_df = pd.DataFrame(policy_fitness_records)
            if len(policy_fitness_df) > 0:
                policy_fitness_table = tabulate(
                    policy_fitness_df,
                    headers=[self.candidate_policy_uri] + list(policy_fitness_df.keys()),
                    tablefmt="grid",
                    # maxcolwidths=25,
                )
                self.logger.info(
                    f"Policy fitness results for candidate policy {self.candidate_policy_uri} "
                    f"and baselines {self.analysis.baseline_policies}:\n"
                    f"{policy_fitness_table}"
                )

        return result_dfs, policy_fitness_records

    @staticmethod
    def get_latest_policy(all_policies, uri):
        if uri in all_policies:
            return uri
        if "wandb" in uri:
            uri = uri.replace("wandb://metta-research/metta/", "")
            uri = uri.replace("wandb://run/", "")
        matching_policies = [i for i in all_policies if uri in i]
        if len(matching_policies) == 0:
            raise ValueError(f"No policy found in DB for candidate policy: {uri}, options are {all_policies}")
        if all([":v" in i for i in matching_policies]):
            matching_policies.sort(key=lambda x: int(x.split(":v")[-1]))
        candidate_uri = matching_policies[-1]

        return candidate_uri

    def policy_fitness(self, metric_data: pd.DataFrame, metric_name: str) -> list[dict]:
        """This returns a list comparing the fitness of the candidate policy (gotten from config or the latest
        checkpoint if called from training) to a collection of baseline policies (optionally specified in config,
        defaulting to all policies evaluated so far in metric_data).
        This expects metric_data to exist for each policy, but this isn't strongly enforced,
        and we'll filter out evaluations that don't have data for every policy we're trying to compare."""

        policy_fitness = []
        if "wandb" in self.candidate_policy_uri:
            uri = self.candidate_policy_uri.replace("wandb://run/", "")
        elif "file" in self.candidate_policy_uri:
            uri = self.candidate_policy_uri.replace("file://", "")
        else:
            uri = self.candidate_policy_uri

        all_policies = metric_data["policy_name"].unique()

        # Get the latest version of the candidate policy
        candidate_uri = self.get_latest_policy(all_policies, uri)

        baseline_policy_names = self.analysis.baseline_policies or all_policies

        # filter by inputted baseline policies if there are, otherwise use all policies as baselines
        baseline_policies = list(set([self.get_latest_policy(all_policies, b) for b in baseline_policy_names]))

        metric_data = metric_data.set_index("policy_name")
        eval_header, npc_header, metric_mean, _ = metric_data.keys()

        evals = metric_data[eval_header].unique()
        npcs = metric_data[npc_header]

        # use a list to index the metric_data to ensure we return a dataframe
        candidate_data: pd.DataFrame = metric_data.loc[[candidate_uri]].set_index(eval_header)
        baseline_data: pd.DataFrame = metric_data.loc[baseline_policies].set_index(eval_header)

        for eval, npc in zip(evals, npcs, strict=False):
            if eval not in candidate_data.index:
                self.logger.info(
                    f"No data found for {eval} in candidate policy {candidate_uri}, cannot compute fitness"
                )
                continue
            if eval not in baseline_data.index:
                self.logger.info(
                    f"No data found for {eval} in baseline policies {baseline_policies}, cannot compute fitness"
                )
                continue
            candidate_mean = candidate_data.loc[eval][metric_mean]
            baseline_mean = np.mean(baseline_data.loc[eval][metric_mean])

            fitness = candidate_mean - baseline_mean

            policy_fitness.append(
                {
                    "eval": eval,
                    "metric": metric_name,
                    "candidate_mean": candidate_mean,
                    "baseline_mean": baseline_mean,
                    "fitness": fitness,
                    "npc_policy_uri": npc,
                }
            )

        return policy_fitness
