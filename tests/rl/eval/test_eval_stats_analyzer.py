from typing import Callable, List

import numpy as np
import pandas as pd
import pytest

from metta.eval.analysis_config import AnalysisConfig, Metric
from metta.sim.eval_stats_analyzer import EvalStatsAnalyzer

# --- Fixtures for shared test resources --- #


@pytest.fixture
def dummy_stats_db():
    """Fixture for a dummy stats DB."""

    class DummyStatsDB:
        def __init__(self, available_metrics, query_return=None):
            self.available_metrics = available_metrics
            self.query_return = query_return

        def query(self, sql_query: str) -> pd.DataFrame:
            if callable(self.query_return):
                return self.query_return(sql_query)
            return self.query_return or pd.DataFrame()

    return DummyStatsDB(available_metrics=["reward_total"])


@pytest.fixture
def dummy_db_for_analyze():
    """Fixture for a dummy DB used in analyze tests."""

    class DummyDBForAnalyze:
        available_metrics = ["reward_total"]

        def query(self, sql_query: str) -> pd.DataFrame:
            # Simulate a simple query result.
            return pd.DataFrame(
                {
                    "policy_name": ["policy1:v1"],
                    "eval": ["eval1"],
                    "npc": "npc",
                    "mean": [10.0],
                    "std": [1.0],
                }
            )

    return DummyDBForAnalyze()


@pytest.fixture
def analysis_config_factory() -> Callable[[List[str]], AnalysisConfig]:
    """Returns a function to create an analysis config with given baseline policies."""

    def _create(baseline_policies):
        return AnalysisConfig(
            metrics=[Metric(metric="reward*")],
            baseline_policies=baseline_policies,
            filters={},
        )

    return _create


@pytest.fixture
def candidate_policy():
    """Fixture for a candidate policy."""
    return "policy1:v1"


@pytest.fixture
def analyzer(dummy_stats_db, analysis_config_factory, candidate_policy):
    """Fixture that creates an analyzer with a given candidate and baseline ["policy1"]."""
    analysis_conf = analysis_config_factory(baseline_policies=["policy1"])
    return EvalStatsAnalyzer(dummy_stats_db, analysis_conf, policy_uri=candidate_policy)


# --- Grouped Tests for the _filters Helper --- #


class TestFilters:
    def test_no_global_or_local(self, analysis_config_factory, dummy_stats_db, candidate_policy):
        analysis_conf = analysis_config_factory(baseline_policies=["policy1"])
        analyzer_instance = EvalStatsAnalyzer(dummy_stats_db, analysis_conf, policy_uri=candidate_policy)
        # With no filters set anywhere, expect None.
        result = analyzer_instance._filters(Metric(metric="reward*"))
        assert result == {}

    def test_global_only(self, analysis_config_factory, dummy_stats_db, candidate_policy):
        analysis_conf = AnalysisConfig(
            metrics=[Metric(metric="reward*")],
            baseline_policies=["policy1"],
            filters={"env": "test_env"},
        )
        analyzer_instance = EvalStatsAnalyzer(dummy_stats_db, analysis_conf, policy_uri=candidate_policy)
        result = analyzer_instance._filters(Metric(metric="reward*"))
        assert result == {"env": "test_env"}

    def test_local_only(self, analysis_config_factory, dummy_stats_db, candidate_policy):
        analysis_conf = analysis_config_factory(baseline_policies=["policy1"])
        analyzer_instance = EvalStatsAnalyzer(dummy_stats_db, analysis_conf, policy_uri=candidate_policy)
        local_item = Metric(metric="reward*", filters={"level": "local_value"})
        result = analyzer_instance._filters(local_item)
        assert result == {"level": "local_value"}

    def test_merging_global_and_local(self, analysis_config_factory, dummy_stats_db, candidate_policy):
        analysis_conf = AnalysisConfig(
            metrics=[Metric(metric="reward*")],
            baseline_policies=["policy1"],
            filters={"env": "test_env", "shared_key": "global_value", "list_key": ["global1", "global2"]},
        )
        analyzer_instance = EvalStatsAnalyzer(dummy_stats_db, analysis_conf, policy_uri=candidate_policy)
        local_item = Metric(
            metric="reward*",
            filters={"level": "local_value", "shared_key": "local_value", "list_key": ["local1", "local2"]},
        )

        result = analyzer_instance._filters(local_item)
        expected = {
            "env": "test_env",
            "level": "local_value",
            "shared_key": "local_value",
            "list_key": ["global1", "global2", "local1", "local2"],
        }
        assert result == expected, f"Expected {result} to be {expected}"


# --- Grouped Tests for get_latest_policy --- #


class TestGetLatestPolicy:
    def test_exact_match(self):
        policies = ["policy1:v1", "policy1:v2", "policy1:v3"]
        result = EvalStatsAnalyzer.get_latest_policy(policies, "policy1:v2")
        assert result == "policy1:v2"

    def test_partial_uri(self):
        policies = ["policy1:v1", "policy1:v2", "policy1:v3"]
        result = EvalStatsAnalyzer.get_latest_policy(policies, "policy1")
        assert result == "policy1:v3"

    def test_with_wandb_prefix(self):
        policies = ["policy1:v1", "policy1:v2", "policy1:v3"]
        candidate = "wandb://run/policy1:v2"
        result = EvalStatsAnalyzer.get_latest_policy(policies, candidate)
        assert result == "policy1:v2"

    def test_no_match(self):
        policies = ["policy1:v1", "policy1:v2"]
        with pytest.raises(ValueError):
            EvalStatsAnalyzer.get_latest_policy(policies, "policy2")


# --- Grouped Tests for policy_fitness --- #


class TestPolicyFitness:
    @pytest.fixture
    def base_analysis_config(self, analysis_config_factory):
        return analysis_config_factory(baseline_policies=["policy1", "policy2"])

    def test_policy_fitness(self, base_analysis_config, dummy_stats_db):
        # Construct a DataFrame with candidate and baseline data.
        df = pd.DataFrame(
            {
                "policy_name": ["policy1:v1", "policy1:v2", "policy2:v1", "policy2:v2"],
                "eval": ["eval1", "eval1", "eval1", "eval1"],
                "npc": "npc",
                "mean": [10.0, 12.0, 8.0, 9.0],
                "std": [1.0, 1.0, 1.0, 1.0],
            }
        )
        analysis_conf = base_analysis_config
        candidate_policy = "policy1:v2"
        analyzer_instance = EvalStatsAnalyzer(dummy_stats_db, analysis_conf, policy_uri=candidate_policy)
        fitness_records = analyzer_instance.policy_fitness(df, "reward_total")
        # For eval1: candidate mean = 12.0; baseline means are 12.0 and 9.0, average = 10.5.
        assert len(fitness_records) == 1
        rec = fitness_records[0]
        assert rec["eval"] == "eval1"
        np.testing.assert_almost_equal(rec["candidate_mean"], 12.0)
        np.testing.assert_almost_equal(rec["baseline_mean"], 10.5)
        np.testing.assert_almost_equal(rec["fitness"], 12.0 - 10.5)

    def test_candidate_missing_eval(self, analysis_config_factory, dummy_stats_db):
        # Candidate only has data for eval1; baseline has data for eval1 and eval2.
        data = pd.DataFrame(
            {
                "policy_name": ["policy1:v1", "policy2:v1", "policy2:v1"],
                "eval": ["eval1", "eval1", "eval2"],
                "npc": "npc",
                "mean": [10.0, 15.0, 16.0],
                "std": [1.0, 1.0, 1.0],
            }
        )
        analysis_conf = analysis_config_factory(baseline_policies=["policy2"])
        candidate_policy = "policy1:v1"
        analyzer_instance = EvalStatsAnalyzer(dummy_stats_db, analysis_conf, policy_uri=candidate_policy)
        fitness_records = analyzer_instance.policy_fitness(data, "reward_total")
        # Only eval1 should be processed.
        assert len(fitness_records) == 1
        rec = fitness_records[0]
        assert rec["eval"] == "eval1"
        np.testing.assert_almost_equal(rec["candidate_mean"], 10.0)
        np.testing.assert_almost_equal(rec["baseline_mean"], 15.0)
        np.testing.assert_almost_equal(rec["fitness"], 10.0 - 15.0)

    def test_baseline_missing_eval(self, analysis_config_factory, dummy_stats_db):
        # Candidate has eval1 and eval2; baseline only has eval1.
        data = pd.DataFrame(
            {
                "policy_name": ["policy1:v1", "policy1:v1", "policy2:v1"],
                "eval": ["eval1", "eval2", "eval1"],
                "npc": "npc",
                "mean": [10.0, 12.0, 15.0],
                "std": [1.0, 1.0, 1.0],
            }
        )
        analysis_conf = analysis_config_factory(baseline_policies=["policy2"])
        candidate_policy = "policy1:v1"
        analyzer_instance = EvalStatsAnalyzer(dummy_stats_db, analysis_conf, policy_uri=candidate_policy)
        fitness_records = analyzer_instance.policy_fitness(data, "reward_total")
        # Only eval1 should be processed.
        assert len(fitness_records) == 1
        rec = fitness_records[0]
        assert rec["eval"] == "eval1"
        np.testing.assert_almost_equal(rec["candidate_mean"], 10.0)
        np.testing.assert_almost_equal(rec["baseline_mean"], 15.0)
        np.testing.assert_almost_equal(rec["fitness"], 10.0 - 15.0)

    def test_candidate_or_baseline_not_in_db(self, analysis_config_factory, dummy_stats_db):
        # Test when the candidate or baseline policy is not present in the data.
        # Here, the candidate "policy1:v1" is missing.
        df_candidate_missing = pd.DataFrame(
            {
                "policy_name": ["policy2:v1", "policy2:v2"],
                "eval": ["eval1", "eval1"],
                "npc": "npc",
                "mean": [15.0, 16.0],
                "std": [1.0, 1.0],
            }
        )
        analysis_conf = analysis_config_factory(baseline_policies=["policy2"])
        candidate_policy = "policy1:v1"
        analyzer_instance = EvalStatsAnalyzer(dummy_stats_db, analysis_conf, policy_uri=candidate_policy)
        with pytest.raises(ValueError) as excinfo:
            analyzer_instance.policy_fitness(df_candidate_missing, "reward_total")
        assert "No policy found in DB for candidate policy" in str(excinfo.value)

        # Now test when the baseline is missing.
        df_baseline_missing = pd.DataFrame(
            {
                "policy_name": ["policy1:v1"],
                "eval": ["eval1"],
                "npc": "npc",
                "mean": [10.0],
                "std": [1.0],
            }
        )
        analysis_conf = analysis_config_factory(baseline_policies=["policy2"])
        candidate_policy = "policy1:v1"
        analyzer_instance = EvalStatsAnalyzer(dummy_stats_db, analysis_conf, policy_uri=candidate_policy)
        with pytest.raises(ValueError) as excinfo:
            analyzer_instance.policy_fitness(df_baseline_missing, "reward_total")
        assert "No policy found in DB for candidate policy" in str(excinfo.value)


# --- Grouped Tests for analyze() --- #


class TestAnalyze:
    def test_no_metrics(self, analysis_config_factory, dummy_stats_db, candidate_policy):
        analysis_conf = AnalysisConfig(
            metrics=[Metric(metric="nonexistent*")],
            baseline_policies=["policy1"],
            filters={},
        )
        analyzer_instance = EvalStatsAnalyzer(dummy_stats_db, analysis_conf, policy_uri=candidate_policy)
        result_dfs, policy_fitness_records = analyzer_instance.analyze()
        assert result_dfs == []
        assert policy_fitness_records == []

    def test_with_data(self, dummy_db_for_analyze, analysis_config_factory, candidate_policy, monkeypatch):
        analysis_conf = AnalysisConfig(
            metrics=[Metric(metric="reward*")],
            baseline_policies=["policy1"],
            filters={},
        )
        analyzer_instance = EvalStatsAnalyzer(dummy_db_for_analyze, analysis_conf, policy_uri=candidate_policy)

        # Replace log_result to confirm it gets called.
        logged = []

        def fake_log(result, metric, filters):
            logged.append((result, metric, filters))

        monkeypatch.setattr(analyzer_instance, "log_result", fake_log)
        monkeypatch.setattr(analyzer_instance, "policy_fitness", lambda metric_data, metric: [])
        result_dfs, policy_fitness_records = analyzer_instance.analyze()
        assert len(result_dfs) == 1
        pd.testing.assert_frame_equal(
            result_dfs[0],
            pd.DataFrame(
                {
                    "policy_name": ["policy1:v1"],
                    "eval": ["eval1"],
                    "npc": "npc",
                    "mean": [10.0],
                    "std": [1.0],
                }
            ),
        )
        assert policy_fitness_records == []
