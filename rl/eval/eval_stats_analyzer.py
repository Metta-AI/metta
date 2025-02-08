import os
from omegaconf import DictConfig, OmegaConf
from typing import List, Optional, Dict, Any
from util.stats_library import MannWhitneyUTest, EloTest, Glicko2Test, get_test_results
from rl.eval.eval_stats_db import EvalStatsDB

class EvalStatsAnalyzer:
    def __init__(
        self,
        stats_db: EvalStatsDB,
        analysis: DictConfig,
        table_name: str,
        **kwargs):
        self.analysis = analysis
        self.stats_db = stats_db
        self.table_name = table_name

    @staticmethod
    def convert_filters_to_sql(filters: Dict[str, Any]) -> Dict[str, Any]:
        """
        Convert config filters to SQL format.

        For example, a value like "> 300" is passed through as-is,
        whereas a plain value is converted to an equality condition.
        """
        sql_filters = {}
        if not filters:
            return sql_filters
        # Convert OmegaConf objects to plain dict/list if necessary
        if OmegaConf.is_config(filters):
            filters = OmegaConf.to_container(filters, resolve=True)

        for key, value in filters.items():
            if isinstance(value, (list, tuple)):
                formatted_values = [f"'{v}'" if isinstance(v, str) else str(v) for v in value]
                sql_filters[key] = f"IN ({', '.join(formatted_values)})"
            elif isinstance(value, str):
                value = value.strip()
                if value.startswith(('>', '<', '=', '!=', '>=', '<=', 'IN', 'BETWEEN', 'IS')):
                    sql_filters[key] = value
                else:
                    sql_filters[key] = f"= '{value}'"
            else:
                sql_filters[key] = f"= {value}"
        return sql_filters

    def prepare_data_for_statistical_tests(self, metrics: List[str], filters: Optional[Dict[str, Any]] = None) -> List[List[dict]]:
        """
        Convert WandB data into the format expected by statistical tests, applying optional filtering.
        This function builds SQL queries manually.
        """
        query = f"SELECT DISTINCT episode_index FROM {self.table_name} ORDER BY episode_index"
        episodes = self.stats_db.query(query)['episode_index'].tolist()

        # Wrap metrics in quotes to handle dot notation in SQL
        metrics = [f'"{metric}"' for metric in metrics]

        # Convert filters to SQL format
        sql_filters = self.convert_filters_to_sql(filters) if filters else {}

        data = []
        for episode_idx in episodes:
            conditions = [f"episode_index = {episode_idx}"]
            for field, value in sql_filters.items():
                conditions.append(f"{field} {value}")
            where_clause = "WHERE " + " AND ".join(conditions)
            episode_query = f"""
                SELECT policy_name, {', '.join(metrics)}
                FROM {self.table_name}
                {where_clause}
            """
            episode_data = self.stats_db.query(episode_query).to_dict('records')
            if episode_data:  # Only include episodes with matching data
                data.append(episode_data)
        return data

    def run_metric_patterns_analysis(self):
        for pattern_config in self.analysis.metric_patterns:
            pattern = pattern_config.pattern
            filters = pattern_config.get('filters', {})
            print(f"\nFinding metrics matching pattern: {pattern}")
            if filters:
                print(f"Using filters: {filters}")
            matched_metrics = self.stats_db.get_metrics_by_pattern(pattern)
            print(f"Found metrics: {matched_metrics}")
            if matched_metrics:
                print(f"Running average_metrics_by_policy for metrics matching '{pattern}':\n")
                result = self.stats_db.average_metrics_by_policy(matched_metrics, filters)
                print(result)

    def run_per_episode_analysis(self):
        for metric_config in self.analysis.per_episode_metrics:
            metric = metric_config.metric
            filters = metric_config.get('filters', {})
            print(f"\nAnalyzing per-episode metric: {metric}")
            if filters:
                print(f"Using filters: {filters}")
            result = self.stats_db.metric_per_episode_per_policy(metric, filters)
            print(f"Per-episode results for {metric}:\n{result}")

    def run_explicit_metrics_analysis(self):
        metrics_list = []
        filters = None
        for metric_config in self.analysis.metrics:
            metrics_list.append(metric_config.metric)
            if metric_config.get('filters'):
                if not filters:
                    filters = metric_config.filters
                elif filters != metric_config.filters:
                    print("Warning: Different filters specified for different metrics. Using first filter.")
        if metrics_list:
            print(f"\nRunning average_metrics_by_policy for metrics: {metrics_list}")
            if filters:
                print(f"Using filters: {filters}")
            result = self.stats_db.average_metrics_by_policy(metrics_list, filters)
            print(result)

    def run_custom_queries(self):
        for query_name, query in self.analysis.queries.items():
            print(f"\nExecuting query: {query_name}\n")
            result = self.stats_db.query(query)
            print(result)

    def run_statistical_tests(self):
        for test_config in self.analysis.statistical_tests:
            test_type = test_config.type
            metrics = test_config.metrics
            mode = test_config.get('mode', 'sum')
            label = test_config.get('label', None)
            scores_path = test_config.get('scores_path', None)
            filters = test_config.get('filters', {})
            print(f"\nRunning {test_type} test for metrics: {metrics}")
            if filters:
                print(f"Using filters: {filters}")

            data = self.prepare_data_for_statistical_tests(metrics, filters)
            if not data:
                print("No data found for the specified metrics and filters")
                continue

            if test_type == "mann_whitney":
                test = MannWhitneyUTest(data, metrics, mode, label)
            elif test_type == "elo":
                test = EloTest(data, metrics, mode, label)
            elif test_type == "glicko2":
                test = Glicko2Test(data, metrics)
            else:
                print(f"Unknown test type: {test_type}")
                continue

            results, formatted_results = get_test_results(test, scores_path)
            print(f"\n{test_type} test results:")
            print(formatted_results)

    def run_all(self):
        if self.analysis.metric_patterns:
            self.run_metric_patterns_analysis()
        if self.analysis.per_episode_metrics:
            self.run_per_episode_analysis()
        if self.analysis.metrics:
            self.run_explicit_metrics_analysis()
        if self.analysis.queries:
            self.run_custom_queries()
        if self.analysis.statistical_tests:
            self.run_statistical_tests()
