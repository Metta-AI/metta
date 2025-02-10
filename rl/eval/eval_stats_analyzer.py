import logging
from omegaconf import DictConfig, OmegaConf
from typing import List, Optional, Dict, Any
from util.stats_library import MannWhitneyUTest, EloTest, Glicko2Test, get_test_results
from rl.eval.eval_stats_db import EvalStatsDB
from tabulate import tabulate


logger = logging.getLogger("eval_stats_analyzer")

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
        self.global_filters = analysis.get('filters', None)

        # Apply global filters to stats_db if they exist
        if self.global_filters is not None:
            self.filter_stats_db()

    def filter_stats_db(self):
        """
        Apply global filters to stats_db if they exist
        """
        where_clause = self.stats_db._build_where_clause(self.global_filters)
        filtered_query = f"""
            SELECT * FROM {self.table_name}
            {where_clause}
        """
        filtered_df = self.stats_db.query(filtered_query)
        self.stats_db = filtered_df

    @staticmethod
    def convert_filters_to_sql(filters: Dict[str, Any]) -> Dict[str, Any]:
        sql_filters = {}
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

    def run_metric_patterns_analysis(self):
        """
        If we passed in metric patterns, i.e. action.use,
        we analyze all metrics that match this pattern """
        for pattern_config in self.analysis.metric_patterns:
            matched_metrics = self.stats_db.get_metrics_by_pattern(pattern_config.pattern)
            filters = pattern_config.get('filters', None)
            if matched_metrics:
                logger.info(f"Analyzing metrics matching '{pattern_config.pattern}':\n")
                result = self.stats_db.average_metrics_by_policy(matched_metrics, filters)
                result_table = tabulate(result, headers=["policy_name"] + list(result.keys()), tablefmt="grid")
                logger.info(result_table)

    def run_per_episode_analysis(self):
        """
        If we passed in per_episode_metrics,
        we analyze the metric per episode per policy
        """
        for metric_config in self.analysis.per_episode_metrics:
            filters = metric_config.get('filters', None)
            result = self.stats_db.metric_per_episode_per_policy(metric_config.metric, filters)
            result_table = tabulate(result, headers=["episode_index"] + list(result.keys()), tablefmt="grid")
            logger.info(f"Per-episode results for {metric_config.metric} with filters {filters}:\n{result_table}")

    def run_explicit_metrics_analysis(self):
        """
        If we passed in metrics, i.e. agent.action.use.altar,
        we analyze the metric per policy, averaging over episodes.
        """
        metrics_list = []
        filters = None
        # if no filters, we display all metrics together
        if not any(metric_config.get('filters', None) for metric_config in self.analysis.metrics):
            metrics_list = [metric_config.metric for metric_config in self.analysis.metrics]
            result = self.stats_db.average_metrics_by_policy(metrics_list, filters)
            result_table = tabulate(result, headers=["policy_name"] + list(result.keys()), tablefmt="grid")
            logger.info(f"Average metrics by policy for metrics: {metrics_list}\n{result_table}")
        # if filters, we display each metric separately
        else:
            for metric_config in self.analysis.metrics:
                filters = metric_config.get('filters', None)
                result = self.stats_db.average_metrics_by_policy([metric_config.metric], filters)
                result_table = tabulate(result, headers=["policy_name"] + list(result.keys()), tablefmt="grid")

                logger.info(f"Average metrics by policy for metric {metric_config.metric} with filters {filters}:\n{result_table}")

    def run_custom_queries(self):
        for query_name, query in self.analysis.queries.items():
            print(f"\nExecuting query: {query_name}\n")
            result = self.stats_db.query(query)
            print(result)

    def prepare_data_for_statistical_tests(self, metrics: List[str], filters: Optional[Dict[str, Any]] = None) -> List[List[dict]]:
        """
        Convert WandB data into the format expected by statistical tests, applying optional filtering.
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
            if episode_data:
                data.append(episode_data)
        if not data:
            logger.warning("No data found for the specified metrics and filters")
        return data

    def run_statistical_tests(self):
        for test_config in self.analysis.statistical_tests:
            test_type = test_config.type
            metrics = test_config.metrics
            mode = test_config.get('mode', 'sum')
            label = test_config.get('label', None)
            scores_path = test_config.get('scores_path', None)
            filters = test_config.get('filters', None)
            print(f"\nRunning {test_type} test for metrics: {metrics} using filters {filters}")

            data = self.prepare_data_for_statistical_tests(metrics, filters)

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


    def run(self):
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
