import logging
from omegaconf import DictConfig, OmegaConf
from typing import Dict, Any, List, Optional
from rl.eval.eval_stats_db import EvalStatsDB
from tabulate import tabulate
import fnmatch
import pandas as pd
from scipy import stats
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

    @staticmethod
    def _significance_test(metrics_df: pd.DataFrame, metric_name: str) -> List[List[str]]:
        """
        Calculates pairwise significance tests between policies for a given metric.
        Uses Mann-Whitney U test (non-parametric) since we can't assume normal distribution.
        """
        policies = metrics_df.keys()
        n_policies = len(policies)
        comparisons = []

        # Calculate pairwise significance
        for i in range(n_policies):
            for j in range(i + 1, n_policies):
                policy1, policy2 = policies[i], policies[j]
                values1 = metrics_df[policy1]
                values2 = metrics_df[policy2]

                # Perform Mann-Whitney U test
                u_statistic, p_value = stats.mannwhitneyu(
                    values1,
                    values2,
                    alternative='two-sided'
                )

                r = 1 - (2 * u_statistic) / (len(values1) * len(values2))

                if r > 0 and p_value < 0.05:
                    interpretation = "pos. effect"
                    p_value_str = colored(f"{p_value:.2f}", 'green')
                    effect_size_str = colored(f"{r:.2f}", 'green')
                elif r < 0 and p_value < 0.05:
                    interpretation = "neg. effect"
                    p_value_str = colored(f"{p_value:.2f}", 'red')
                    effect_size_str = colored(f"{r:.2f}", 'red')
                else:
                    interpretation = "no effect"
                    p_value_str = f"{p_value:.2f}" if p_value is not None else "N/A"
                    effect_size_str = f"{r:.2f}" if r is not None else "N/A"

                comparisons.append([
                    policy1[:5] + '...' + policy1[-20:] if len(policy1) > 25 else policy1,
                    policy2[:5] + '...' + policy2[-20:] if len(policy2) > 25 else policy2,
                    p_value_str,
                    effect_size_str,
                    interpretation,
                    metric_name
                ])

        return comparisons


    def _analyze_metrics(self, metric_fields: List[str], filters: Optional[Dict[str, Any]] = None, group_by_episode: bool = False) -> pd.DataFrame:
        result_dfs = []
        significance_results = []
        for metric in metric_fields:
            if not metric in self.stats_db.available_metrics:
                logger.info(f"Metric {metric} not found in stats_db")
                continue
            df_per_episode = self.stats_db._metric(metric, filters)
            if not group_by_episode:
                mean_series = df_per_episode.mean(axis=0)
                std_series = df_per_episode.std(axis=0)
                metric_df = pd.DataFrame({
                    f'{metric}_mean': mean_series,
                    f'{metric}_std': std_series
                })
                result_dfs.append(metric_df)
            else:
                result_dfs.append(df_per_episode)

            # Only calculate significance if there are at least 2 policies
            if df_per_episode.shape[1] > 1:
                significance_results += self._significance_test(df_per_episode, metric)

        metrics_df = pd.concat(result_dfs, axis=1) if len(result_dfs) > 0 else None

        return metrics_df, significance_results

    def analyze(self):
        metric_configs = {}
        for cfg in self.analysis.metrics:
            metric = cfg.metric
            if '*' in metric or '?' in metric:
                # Use fnmatch.filter to match against the available metrics
                metric_configs[cfg] = fnmatch.filter(self.stats_db.available_metrics, metric)
            else:
                metric_configs[cfg] = [metric]

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

        return results, significances
