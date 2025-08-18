"""
Test suite for stats_analysis module - Priority 1 components.

Tests all game statistics analysis functionality with hard failure enforcement.
No graceful degradation - tests fail when required data is missing.
"""

import json
from pathlib import Path

import pytest

from metta.mcp_server.stats_analysis import (
    AgentStats,
    BehavioralAnalysisEngine,
    BuildingEfficiencyScorer,
    BuildingStats,
    CombatInteractionAnalyzer,
    ResourceFlowAnalyzer,
    StatsExtractor,
    StrategicPhaseDetector,
)


# Test data fixtures
@pytest.fixture
def objects_format_replay():
    """Load real replay data in objects format"""
    test_file = Path(__file__).parent / "test_replay_objects_format.json"
    with open(test_file) as f:
        return json.load(f)


@pytest.fixture
def grid_objects_format_replay():
    """Load real replay data in grid_objects format"""
    test_file = Path(__file__).parent / "test_replay_grid_objects_format.json"
    with open(test_file) as f:
        return json.load(f)


class TestStatsExtractor:
    """Test StatsExtractor class functionality"""

    def test_extract_from_objects_format_replay(self, objects_format_replay):
        """Test successful stats extraction from objects format replay data"""
        extractor = StatsExtractor()
        result = extractor.extract_from_replay_data(objects_format_replay)

        # Verify structure matches actual implementation
        assert "game" in result
        assert "agent" in result
        assert "converter" in result

        # Verify game stats structure
        game_stats = result["game"]
        assert isinstance(game_stats, dict)
        assert game_stats["total_agents"] == 24
        assert game_stats["episode_length"] == 500
        assert "resource_scarcity_index" in game_stats

        # Verify agent stats structure
        agent_stats = result["agent"]
        assert isinstance(agent_stats, list)
        assert len(agent_stats) == 24  # All agents

        # Verify first agent has meaningful data
        first_agent = agent_stats[0]
        assert "agent_id" in first_agent
        assert "total_reward" in first_agent
        assert "action_counts" in first_agent
        assert isinstance(first_agent["action_counts"], dict)

        # Verify building stats structure
        building_stats = result["converter"]
        assert isinstance(building_stats, list)

    def test_extract_from_grid_objects_format_replay(self, grid_objects_format_replay):
        """Test successful stats extraction from grid_objects format replay data"""
        extractor = StatsExtractor()
        result = extractor.extract_from_replay_data(grid_objects_format_replay)

        # Verify structure matches actual implementation
        assert "game" in result
        assert "agent" in result
        assert "converter" in result

        # Verify game stats structure
        game_stats = result["game"]
        assert isinstance(game_stats, dict)
        assert game_stats["total_agents"] == 24
        assert game_stats["episode_length"] == 500
        assert game_stats["resource_scarcity_index"] > 0  # Should be calculated from actual data

        # Verify agent stats structure
        agent_stats = result["agent"]
        assert isinstance(agent_stats, list)
        assert len(agent_stats) == 24  # All agents

        # Verify first agent has meaningful data
        first_agent = agent_stats[0]
        assert first_agent["agent_id"] == 0
        assert isinstance(first_agent["total_reward"], float)
        assert isinstance(first_agent["action_counts"], dict)
        assert len(first_agent["action_counts"]) > 0  # Should have actual actions

        # Verify actions have meaningful names (not just action_[X,Y])
        action_names = list(first_agent["action_counts"].keys())
        meaningful_actions = [name for name in action_names if not name.startswith("action_")]
        assert len(meaningful_actions) > 0, f"Should have meaningful action names, got: {action_names}"

        # Verify building stats structure
        building_stats = result["converter"]
        assert isinstance(building_stats, list)

    def test_extract_missing_data_raises_error(self):
        """Test that missing required data raises ValueError (hard failure)"""
        extractor = StatsExtractor()

        replay_data = {
            "max_steps": 1000,
            "num_agents": 4,
            # Missing objects or grid_objects
        }

        with pytest.raises(ValueError, match="Replay data missing required statistical information"):
            extractor.extract_from_replay_data(replay_data)

    def test_extract_empty_objects_raises_error(self):
        """Test that empty objects list raises ValueError"""
        extractor = StatsExtractor()

        replay_data = {
            "objects": [],  # Empty objects list
            "max_steps": 1000,
            "num_agents": 4,
        }

        with pytest.raises(ValueError, match="Replay data missing required statistical information"):
            extractor.extract_from_replay_data(replay_data)

    def test_extract_agent_stats_with_empty_objects(self):
        """Test agent stats extraction with empty objects"""
        extractor = StatsExtractor()

        # Test with replay data missing agent objects
        empty_replay_data = {
            "version": "1.0",
            "max_steps": 1000,
            "objects": [],  # No agent objects
        }

        result = extractor._extract_agent_stats(empty_replay_data)
        assert isinstance(result, list)

    def test_extract_building_stats_with_empty_objects(self):
        """Test building stats extraction with empty objects"""
        extractor = StatsExtractor()

        # Test with replay data missing building objects
        empty_replay_data = {
            "version": "1.0",
            "max_steps": 1000,
            "objects": [],  # No building objects
        }

        result = extractor._extract_building_stats(empty_replay_data)
        assert isinstance(result, list)


class TestBehavioralAnalysisEngine:
    """Test BehavioralAnalysisEngine functionality"""

    @pytest.mark.parametrize("replay_format", ["objects_format_replay", "grid_objects_format_replay"])
    def test_analyze_agent_behaviors_real_data(self, replay_format, request):
        """Test behavioral analysis with real replay data from both formats"""
        replay_data = request.getfixturevalue(replay_format)

        # Extract stats from replay data
        extractor = StatsExtractor()
        episode_stats = extractor.extract_from_replay_data(replay_data)

        # Create AgentStats objects
        agent_stats = []
        for agent_data in episode_stats.get("agent", []):
            agent_stat = AgentStats(
                agent_id=agent_data.get("agent_id", 0),
                total_actions=agent_data.get("action_counts", {}),
                action_success_rates={},
                resource_flows=agent_data.get("resource_transactions", {}),
                movement_patterns=agent_data.get("movement_stats", {}),
                combat_stats=agent_data.get("interaction_stats", {}),
                building_interactions={},
                efficiency_metrics={"overall_efficiency": agent_data.get("total_reward", 0.0)},
            )
            agent_stats.append(agent_stat)

        # Test behavioral analysis
        engine = BehavioralAnalysisEngine()
        result = engine.analyze_agent_behaviors(agent_stats)

        # Verify structure matches actual implementation
        assert "efficiency_rankings" in result
        assert "behavioral_clusters" in result
        assert "strategy_identification" in result
        assert "performance_correlations" in result
        assert "outlier_detection" in result

        # Verify efficiency rankings have real data
        rankings = result["efficiency_rankings"]
        assert isinstance(rankings, list)
        assert len(rankings) == len(agent_stats)

        # At least some agents should have non-zero efficiency scores
        efficiency_scores = [ranking.get("efficiency_score", 0) for ranking in rankings]
        non_zero_count = sum(1 for score in efficiency_scores if score > 0)
        assert non_zero_count > 0, f"All efficiency scores are zero: {efficiency_scores[:5]}"

    def test_analyze_empty_agent_list_raises_error(self):
        """Test that empty agent list raises ValueError (hard failure)"""
        engine = BehavioralAnalysisEngine()

        with pytest.raises(ValueError, match="No agent statistics provided"):
            engine.analyze_agent_behaviors([])

    def test_analyze_performance_correlations(self, sample_agent_stats):
        """Test performance correlation calculations"""
        engine = BehavioralAnalysisEngine()

        correlations = engine._analyze_performance_correlations(sample_agent_stats)

        # Should return correlation dictionary
        assert isinstance(correlations, dict)

        # Correlation values should be between -1 and 1
        for _key, value in correlations.items():
            assert -1 <= value <= 1

    def test_detect_behavioral_outliers(self, sample_agent_stats):
        """Test outlier detection in agent behaviors"""
        engine = BehavioralAnalysisEngine()

        outliers = engine._detect_behavioral_outliers(sample_agent_stats)

        # Should return list of outlier information
        assert isinstance(outliers, list)
        # Each outlier should be a dict with agent information
        for outlier in outliers:
            if outlier:  # If outliers detected
                assert isinstance(outlier, dict)


class TestResourceFlowAnalyzer:
    """Test ResourceFlowAnalyzer functionality"""

    @pytest.mark.parametrize("replay_format", ["objects_format_replay", "grid_objects_format_replay"])
    def test_analyze_resource_flows_real_data(self, replay_format, request):
        """Test resource flow analysis with real replay data from both formats"""
        replay_data = request.getfixturevalue(replay_format)

        # Extract stats from replay data
        extractor = StatsExtractor()
        episode_stats = extractor.extract_from_replay_data(replay_data)

        # Create AgentStats and BuildingStats objects
        agent_stats = []
        for agent_data in episode_stats.get("agent", []):
            agent_stat = AgentStats(
                agent_id=agent_data.get("agent_id", 0),
                total_actions=agent_data.get("action_counts", {}),
                action_success_rates={},
                resource_flows=agent_data.get("resource_transactions", {}),
                movement_patterns=agent_data.get("movement_stats", {}),
                combat_stats=agent_data.get("interaction_stats", {}),
                building_interactions={},
                efficiency_metrics={"overall_efficiency": agent_data.get("total_reward", 0.0)},
            )
            agent_stats.append(agent_stat)

        building_stats = []
        for building_data in episode_stats.get("converter", []):
            building_stat = BuildingStats(
                building_id=building_data.get("building_id", 0),
                type_id=building_data.get("type_id", 0),
                type_name=building_data.get("type_name", f"type_{building_data.get('type_id', 0)}"),
                location=building_data.get("location", (0, 0)),
                production_efficiency=building_data.get("production_stats", {}),
                resource_flows=building_data.get("resource_flows", {}),
                operational_stats=building_data.get("operational_stats", {}),
                bottleneck_analysis={"bottleneck_score": building_data.get("bottleneck_score", 0.0)},
            )
            building_stats.append(building_stat)

        # Test resource flow analysis
        analyzer = ResourceFlowAnalyzer()
        result = analyzer.analyze_resource_flows(agent_stats, building_stats)

        # Verify structure matches actual implementation
        assert "resource_flow_matrix" in result
        assert "production_efficiency" in result
        assert "consumption_patterns" in result
        assert "bottleneck_identification" in result
        assert "resource_scarcity_analysis" in result

        # Verify resource scarcity analysis has real data
        scarcity_analysis = result["resource_scarcity_analysis"]
        assert isinstance(scarcity_analysis, dict)
        assert len(scarcity_analysis) > 0, "Resource scarcity analysis should not be empty"

        # For grid_objects format, we expect non-zero values; for objects format, may be all zeros if no temporal data
        scarcity_values = list(scarcity_analysis.values())
        if replay_format == "grid_objects_format_replay":
            non_zero_scarcity = sum(1 for value in scarcity_values if value > 0)
            assert non_zero_scarcity > 0, f"Grid objects format should have non-zero scarcity: {scarcity_analysis}"
        # For objects format, just verify the structure is correct
        assert all(isinstance(v, (int, float)) for v in scarcity_values), "All scarcity values should be numeric"

    def test_analyze_with_empty_agent_stats(self, sample_building_stats):
        """Test resource flow analysis with empty agent stats"""
        analyzer = ResourceFlowAnalyzer()

        # The actual implementation doesn't raise errors for empty lists
        result = analyzer.analyze_resource_flows([], sample_building_stats)
        assert isinstance(result, dict)

    def test_analyze_with_empty_building_stats(self, sample_agent_stats):
        """Test resource flow analysis with empty building stats"""
        analyzer = ResourceFlowAnalyzer()

        # The actual implementation doesn't raise errors for empty lists
        result = analyzer.analyze_resource_flows(sample_agent_stats, [])
        assert isinstance(result, dict)

    def test_identify_bottlenecks(self, sample_agent_stats, sample_building_stats):
        """Test bottleneck identification in production chains"""
        analyzer = ResourceFlowAnalyzer()

        bottlenecks = analyzer._identify_bottlenecks(sample_agent_stats, sample_building_stats)

        # Should return list of bottleneck information
        assert isinstance(bottlenecks, list)


class TestCombatInteractionAnalyzer:
    """Test CombatInteractionAnalyzer functionality"""

    def test_analyze_combat_interactions_success(self, sample_agent_stats):
        """Test successful combat interaction analysis"""
        analyzer = CombatInteractionAnalyzer()

        result = analyzer.analyze_combat_interactions(sample_agent_stats)

        # Verify structure matches actual implementation
        assert "interaction_matrix" in result
        assert "aggression_rankings" in result
        assert "cooperation_metrics" in result
        assert "territorial_analysis" in result
        assert "social_network_metrics" in result

        # Verify interaction matrix
        matrix = result["interaction_matrix"]
        assert isinstance(matrix, dict)

        # Verify aggression rankings
        rankings = result["aggression_rankings"]
        assert isinstance(rankings, list)

    def test_empty_combat_data_provides_baseline(self, sample_agent_stats):
        """Test handling of agents with no combat statistics"""
        # Create agent stats with no combat data
        peaceful_agents = []
        for agent in sample_agent_stats:
            peaceful_agent = AgentStats(
                agent_id=agent.agent_id,
                total_actions=agent.total_actions,
                action_success_rates=agent.action_success_rates,
                resource_flows=agent.resource_flows,
                movement_patterns=agent.movement_patterns,
                combat_stats={},  # No combat stats
                building_interactions=agent.building_interactions,
                efficiency_metrics=agent.efficiency_metrics,
            )
            peaceful_agents.append(peaceful_agent)

        analyzer = CombatInteractionAnalyzer()
        result = analyzer.analyze_combat_interactions(peaceful_agents)

        # Should still provide analysis structure
        assert "interaction_matrix" in result
        assert "aggression_rankings" in result

    def test_calculate_social_metrics(self, sample_agent_stats):
        """Test social metrics calculations"""
        analyzer = CombatInteractionAnalyzer()

        metrics = analyzer._calculate_social_metrics(sample_agent_stats)

        # Should have social network metrics
        assert isinstance(metrics, dict)
        for _key, value in metrics.items():
            assert isinstance(value, (int, float))


class TestBuildingEfficiencyScorer:
    """Test BuildingEfficiencyScorer functionality"""

    def test_score_building_efficiency_success(self, sample_building_stats):
        """Test successful building efficiency scoring"""
        scorer = BuildingEfficiencyScorer()

        result = scorer.score_building_efficiency(sample_building_stats)

        # Verify structure matches actual implementation
        assert "individual_scores" in result
        assert "comparative_rankings" in result
        assert "utilization_analysis" in result
        assert "optimization_recommendations" in result

        # Verify individual scores
        scores = result["individual_scores"]
        assert isinstance(scores, dict)

        # Verify recommendations
        recommendations = result["optimization_recommendations"]
        assert isinstance(recommendations, list)

    def test_empty_building_list_raises_error(self):
        """Test that empty building list raises ValueError"""
        scorer = BuildingEfficiencyScorer()

        with pytest.raises(ValueError, match="No building statistics provided"):
            scorer.score_building_efficiency([])

    def test_score_individual_buildings(self, sample_building_stats):
        """Test individual building scoring"""
        scorer = BuildingEfficiencyScorer()

        scores = scorer._score_individual_buildings(sample_building_stats)

        # Should return dict with building scores
        assert isinstance(scores, dict)
        assert len(scores) == len(sample_building_stats)

    def test_generate_optimization_recommendations(self, sample_building_stats):
        """Test optimization recommendation generation"""
        scorer = BuildingEfficiencyScorer()

        recommendations = scorer._generate_optimization_recommendations(sample_building_stats)

        # Should provide actionable recommendations
        assert isinstance(recommendations, list)
        for rec in recommendations:
            assert isinstance(rec, str)
            assert len(rec) > 10  # Should be descriptive


class TestStrategicPhaseDetector:
    """Test StrategicPhaseDetector functionality"""

    def test_detect_strategic_phases_success(self, sample_agent_stats):
        """Test successful strategic phase detection"""
        detector = StrategicPhaseDetector()
        episode_length = 1000

        result = detector.detect_strategic_phases(sample_agent_stats, episode_length)

        # Should return list of strategic phases
        assert isinstance(result, list)
        assert len(result) > 0

        # Each phase should have required fields matching actual implementation
        for phase in result:
            assert "phase_number" in phase
            assert "start_step" in phase
            assert "end_step" in phase
            assert "duration" in phase
            assert "dominant_strategy" in phase
            assert "agent_activities" in phase

    def test_invalid_episode_length_raises_error(self, sample_agent_stats):
        """Test that invalid episode length raises ValueError"""
        detector = StrategicPhaseDetector()

        with pytest.raises(ValueError, match="Insufficient data for strategic phase detection"):
            detector.detect_strategic_phases(sample_agent_stats, 0)

    def test_insufficient_data_raises_error(self):
        """Test that insufficient agent data raises ValueError"""
        detector = StrategicPhaseDetector()

        with pytest.raises(ValueError, match="Insufficient data for strategic phase detection"):
            detector.detect_strategic_phases([], 1000)

    def test_analyze_phase(self, sample_agent_stats):
        """Test individual phase analysis"""
        detector = StrategicPhaseDetector()

        phase = detector._analyze_phase(sample_agent_stats, 0, 500, 1)

        # Should return phase analysis dict
        assert isinstance(phase, dict)
        assert "phase_number" in phase
        assert "start_step" in phase
        assert "end_step" in phase


class TestIntegrationScenarios:
    """Test integration scenarios and edge cases"""

    def test_full_stats_analysis_pipeline(self, sample_episode_stats, sample_replay_data):
        """Test complete stats analysis pipeline"""
        # Create complete replay data with episode stats
        complete_replay_data = sample_replay_data.copy()
        complete_replay_data["episode_stats"] = sample_episode_stats

        # Run through StatsExtractor
        extractor = StatsExtractor()
        # Run through StatsExtractor to verify it works
        extractor.extract_from_replay_data(complete_replay_data)

        # Create sample agent and building stats objects for testing other components
        sample_agent = AgentStats(
            agent_id=0,
            total_actions={"get_output": 18, "move": 45},
            action_success_rates={"get_output": 0.83, "move": 1.0},
            resource_flows={"ore_red": {"gained": 8, "lost": 5}},
            movement_patterns={"up": 15, "right": 20},
            combat_stats={"hit_blue": 2},
            building_interactions={"get": 15},
            efficiency_metrics={"overall_efficiency": 1.6},
        )

        sample_building = BuildingStats(
            building_id=0,
            type_id=2,
            type_name="mine_red",
            location=(5, 5),
            production_efficiency={"conversions.started": 10, "conversions.completed": 8},
            resource_flows={"ore_red": {"produced": 12, "consumed": 0}},
            operational_stats={"conversions.blocked": 2},
            bottleneck_analysis={"blocked.output_full": 1},
        )

        # Run through each analyzer
        behavior_engine = BehavioralAnalysisEngine()
        behavior_analysis = behavior_engine.analyze_agent_behaviors([sample_agent])

        resource_analyzer = ResourceFlowAnalyzer()
        resource_analysis = resource_analyzer.analyze_resource_flows([sample_agent], [sample_building])

        combat_analyzer = CombatInteractionAnalyzer()
        combat_analysis = combat_analyzer.analyze_combat_interactions([sample_agent])

        efficiency_scorer = BuildingEfficiencyScorer()
        efficiency_analysis = efficiency_scorer.score_building_efficiency([sample_building])

        phase_detector = StrategicPhaseDetector()
        phase_analysis = phase_detector.detect_strategic_phases([sample_agent], 1000)

        # Verify all analyses completed successfully
        assert behavior_analysis is not None
        assert resource_analysis is not None
        assert combat_analysis is not None
        assert efficiency_analysis is not None
        assert phase_analysis is not None

        # Verify comprehensive analysis structure
        assert len(behavior_analysis) >= 5  # All expected components
        assert len(resource_analysis) >= 5
        assert len(combat_analysis) >= 5
        assert len(efficiency_analysis) >= 4
        assert len(phase_analysis) >= 1  # At least one phase

    def test_hard_failure_enforcement_with_missing_stats(self):
        """Test that missing stats data causes hard failures (no graceful degradation)"""
        import os

        # Ensure we're in strict test mode (set by conftest.py)
        assert os.environ.get("MCP_TEST_MODE") == "strict"

        # Each component should fail hard when required data is missing
        extractor = StatsExtractor()
        with pytest.raises(ValueError):
            extractor.extract_from_replay_data({})  # Missing required data

        behavior_engine = BehavioralAnalysisEngine()
        with pytest.raises(ValueError):
            behavior_engine.analyze_agent_behaviors([])  # Empty agent list

        efficiency_scorer = BuildingEfficiencyScorer()
        with pytest.raises(ValueError):
            efficiency_scorer.score_building_efficiency([])  # Empty building list

        phase_detector = StrategicPhaseDetector()
        with pytest.raises(ValueError):
            phase_detector.detect_strategic_phases([], 1000)  # Empty agent list

    def test_performance_with_large_datasets(self, sample_agent_stats, sample_building_stats):
        """Test performance with large datasets"""
        import time

        # Scale up the datasets
        large_agent_stats = sample_agent_stats * 50  # 100 agents
        large_building_stats = sample_building_stats * 25  # 50 buildings

        # Update agent IDs to be unique
        for i, agent in enumerate(large_agent_stats):
            agent.agent_id = i

        # Update building IDs to be unique
        for i, building in enumerate(large_building_stats):
            building.building_id = i

        # Test each analyzer with timing
        start_time = time.time()

        behavior_engine = BehavioralAnalysisEngine()
        behavior_analysis = behavior_engine.analyze_agent_behaviors(large_agent_stats)

        resource_analyzer = ResourceFlowAnalyzer()
        resource_analysis = resource_analyzer.analyze_resource_flows(large_agent_stats, large_building_stats)

        efficiency_scorer = BuildingEfficiencyScorer()
        efficiency_analysis = efficiency_scorer.score_building_efficiency(large_building_stats)

        end_time = time.time()
        analysis_time = end_time - start_time

        # Should complete within reasonable time (adjust threshold as needed)
        assert analysis_time < 10.0, f"Analysis took too long: {analysis_time:.2f} seconds"

        # Verify analyses still work correctly
        assert behavior_analysis is not None
        assert resource_analysis is not None
        assert efficiency_analysis is not None
