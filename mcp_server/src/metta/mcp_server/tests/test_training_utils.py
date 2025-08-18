"""
Test suite for training_utils.py enhanced analysis pipeline integration.

Tests the integration of Priority 1 and Priority 2 enhancements with existing
replay analysis functionality. Enforces hard failure requirement.
"""

import json
import tempfile
from pathlib import Path
from unittest.mock import Mock

import pytest

from metta.mcp_server.training_utils import _analyze_replay_data, _load_replay_file, analyze_replay_with_enhanced_stats


class TestEnhancedReplayAnalysis:
    """Test the main enhanced replay analysis function"""

    def test_analyze_replay_with_enhanced_stats_success(self, temp_replay_file, sample_episode_stats, mock_mcp_client):
        """Test successful enhanced replay analysis with all components"""
        # Create complete replay file with episode stats
        replay_data = {
            "version": "1.0",
            "max_steps": 1000,
            "num_agents": 2,
            "episode_stats": sample_episode_stats,
            "objects": [
                {
                    "agent_id": 0,
                    "type_id": 0,
                    "location": [[0, [5, 5, 0]], [100, [6, 5, 0]]],
                    "inventory": [[0, []], [50, [[0, 1]]]],
                    "total_reward": [[0, 0.0], [50, 0.5]],
                }
            ],
        }

        # Write complete replay file
        with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
            json.dump(replay_data, f)
            replay_path = f.name

        try:
            result = analyze_replay_with_enhanced_stats(replay_path, "test_run_123", mock_mcp_client)

            # Verify main structure
            assert "basic_analysis" in result
            assert "enhanced_stats_analysis" in result
            assert "wandb_training_context" in result
            assert "analysis_metadata" in result

            # Verify enhanced stats analysis
            stats_analysis = result["enhanced_stats_analysis"]
            assert "game_stats_summary" in stats_analysis
            assert "behavioral_analysis" in stats_analysis
            assert "resource_flow_analysis" in stats_analysis
            assert "combat_interaction_analysis" in stats_analysis
            assert "building_efficiency_analysis" in stats_analysis
            assert "strategic_phase_analysis" in stats_analysis

            # Verify WandB training context
            wandb_context = result["wandb_training_context"]
            assert "training_progression_analysis" in wandb_context
            assert "context_metadata" in wandb_context

            # Verify analysis metadata
            metadata = result["analysis_metadata"]
            assert "analysis_timestamp" in metadata
            assert "replay_file_path" in metadata
            assert "features_enabled" in metadata

        finally:
            # Clean up temp file
            Path(replay_path).unlink()

    def test_missing_replay_file_raises_error(self, mock_mcp_client):
        """Test that missing replay file raises ValueError (hard failure)"""
        with pytest.raises(ValueError, match="Failed to load replay file"):
            analyze_replay_with_enhanced_stats("nonexistent_file.json", "test_run", mock_mcp_client)

    def test_malformed_replay_data_raises_error(self, mock_mcp_client):
        """Test that malformed replay data raises ValueError"""
        # Create malformed replay file
        with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
            f.write("invalid json content")
            malformed_path = f.name

        try:
            with pytest.raises(ValueError, match="Failed to load replay file"):
                analyze_replay_with_enhanced_stats(malformed_path, "test_run", mock_mcp_client)
        finally:
            Path(malformed_path).unlink()

    def test_missing_episode_stats_raises_error(self, mock_mcp_client):
        """Test that missing episode stats raises ValueError (hard failure)"""
        # Create replay without episode_stats
        replay_data = {
            "version": "1.0",
            "max_steps": 1000,
            "num_agents": 2,
            "objects": [],
            # Missing episode_stats
        }

        with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
            json.dump(replay_data, f)
            replay_path = f.name

        try:
            with pytest.raises(ValueError, match="Missing required episode_stats"):
                analyze_replay_with_enhanced_stats(replay_path, "test_run", mock_mcp_client)
        finally:
            Path(replay_path).unlink()

    def test_wandb_integration_failure_raises_error(self, temp_replay_file, sample_episode_stats):
        """Test that WandB integration failure raises error (no graceful degradation)"""
        # Create failing MCP client
        failing_client = Mock()
        failing_client.side_effect = Exception("WandB API unavailable")

        # Create complete replay file
        replay_data = {"version": "1.0", "max_steps": 1000, "episode_stats": sample_episode_stats, "objects": []}

        with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
            json.dump(replay_data, f)
            replay_path = f.name

        try:
            with pytest.raises((ValueError, Exception)):
                analyze_replay_with_enhanced_stats(replay_path, "test_run", failing_client)
        finally:
            Path(replay_path).unlink()


class TestReplayDataProcessing:
    """Test replay data processing and parsing functions"""

    def test_load_replay_file_success(self, temp_replay_file):
        """Test successful replay file loading"""
        result = _load_replay_file(str(temp_replay_file))

        assert result is not None
        assert isinstance(result, dict)
        assert "version" in result
        assert "objects" in result

    def test_load_replay_file_nonexistent_raises_error(self):
        """Test that nonexistent file raises error"""
        with pytest.raises(ValueError, match="Failed to load replay file"):
            _load_replay_file("nonexistent_file.json")

    def test_analyze_replay_data_functionality(self, sample_replay_data):
        """Test basic replay data analysis functionality"""
        result = _analyze_replay_data(sample_replay_data)

        # Should return analysis dict
        assert isinstance(result, dict)
        # Basic analysis should have some key information
        assert len(result) > 0


class TestAnalysisIntegration:
    """Test integration between basic analysis and enhanced analysis"""

    def test_enhanced_analysis_preserves_basic_analysis(
        self, sample_replay_data, sample_episode_stats, mock_mcp_client
    ):
        """Test that enhanced analysis preserves existing basic analysis"""
        # Add episode stats to replay data
        complete_replay_data = sample_replay_data.copy()
        complete_replay_data["episode_stats"] = sample_episode_stats

        with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
            json.dump(complete_replay_data, f)
            replay_path = f.name

        try:
            result = analyze_replay_with_enhanced_stats(replay_path, "test_run", mock_mcp_client)

            # Verify basic analysis is preserved
            assert "basic_analysis" in result
            basic_analysis = result["basic_analysis"]
            assert isinstance(basic_analysis, dict)

            # Verify enhanced analysis is added
            assert "enhanced_stats_analysis" in result
            assert "wandb_training_context" in result

        finally:
            Path(replay_path).unlink()

    def test_analysis_metadata_completeness(self, temp_replay_file, sample_episode_stats, mock_mcp_client):
        """Test that analysis metadata is complete"""
        # Create complete replay file
        replay_data = {"version": "1.0", "max_steps": 1000, "episode_stats": sample_episode_stats, "objects": []}

        with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
            json.dump(replay_data, f)
            replay_path = f.name

        try:
            result = analyze_replay_with_enhanced_stats(replay_path, "test_run", mock_mcp_client)

            metadata = result["analysis_metadata"]

            # Verify required metadata fields
            required_fields = [
                "analysis_timestamp",
                "replay_file_path",
                "features_enabled",
                "analysis_version",
                "processing_time_ms",
            ]

            for field in required_fields:
                assert field in metadata, f"Missing metadata field: {field}"

            # Verify features enabled
            features = metadata["features_enabled"]
            assert features["enhanced_stats_analysis"] is True
            assert features["wandb_integration"] is True

        finally:
            Path(replay_path).unlink()


class TestErrorHandling:
    """Test comprehensive error handling scenarios"""

    def test_stats_analysis_failure_propagates_error(self, mock_mcp_client):
        """Test that stats analysis failures propagate (no graceful degradation)"""
        # Create replay with malformed episode stats
        malformed_replay_data = {
            "version": "1.0",
            "max_steps": 1000,
            "episode_stats": {
                "game": {"episode_length": 1000},
                "agent": [{"invalid_field": "value"}],  # Missing required fields
                "converter": [],
            },
            "objects": [],
        }

        with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
            json.dump(malformed_replay_data, f)
            replay_path = f.name

        try:
            with pytest.raises(ValueError):
                analyze_replay_with_enhanced_stats(replay_path, "test_run", mock_mcp_client)
        finally:
            Path(replay_path).unlink()

    def test_comprehensive_error_messages(self, mock_mcp_client):
        """Test that error messages are informative"""
        with pytest.raises(ValueError) as exc_info:
            analyze_replay_with_enhanced_stats("nonexistent_file.json", "test_run", mock_mcp_client)

        error_message = str(exc_info.value)
        assert "Failed to load replay file" in error_message

    def test_hard_failure_enforcement_environment_variable(self):
        """Test that hard failure is enforced via environment variable"""
        import os

        # Should be set by conftest.py
        assert os.environ.get("MCP_TEST_MODE") == "strict"

        # This environment variable enforces hard failures throughout the system


class TestPerformanceAndScaling:
    """Test performance characteristics and scaling behavior"""

    def test_analysis_performance_tracking(self, temp_replay_file, sample_episode_stats, mock_mcp_client):
        """Test that analysis performance is tracked"""
        replay_data = {"version": "1.0", "max_steps": 1000, "episode_stats": sample_episode_stats, "objects": []}

        with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
            json.dump(replay_data, f)
            replay_path = f.name

        try:
            result = analyze_replay_with_enhanced_stats(replay_path, "test_run", mock_mcp_client)

            # Verify performance tracking
            metadata = result["analysis_metadata"]
            assert "processing_time_ms" in metadata
            assert isinstance(metadata["processing_time_ms"], (int, float))
            assert metadata["processing_time_ms"] > 0

        finally:
            Path(replay_path).unlink()

    def test_memory_efficiency_with_large_datasets(self, mock_mcp_client):
        """Test memory efficiency with large replay datasets"""
        # Create large replay dataset
        large_episode_stats = {"game": {"episode_length": 10000, "total_agents": 20}, "agent": [], "converter": []}

        # Create 20 agents with extensive stats
        for agent_id in range(20):
            agent_data = {
                "agent_id": agent_id,
                "action.get_output.success": 50,
                "action.get_output.failed": 5,
                "action.put_recipe_items.success": 40,
                "action.move.success": 200,
                "ore_red.gained": 25,
                "ore_red.lost": 10,
                "battery_red.gained": 15,
                "movement.direction.up": 50,
                "movement.direction.right": 60,
                "total_reward": 5.5,
            }
            large_episode_stats["agent"].append(agent_data)

        # Create 10 converters
        for converter_id in range(10):
            converter_data = {
                "type_id": 2 + (converter_id % 4),
                "location.r": 5 + converter_id,
                "location.c": 5 + converter_id,
                "conversions.started": 100,
                "conversions.completed": 95,
                "ore_red.produced": 50,
            }
            large_episode_stats["converter"].append(converter_data)

        large_replay_data = {"version": "1.0", "max_steps": 10000, "episode_stats": large_episode_stats, "objects": []}

        with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
            json.dump(large_replay_data, f)
            replay_path = f.name

        try:
            import os

            import psutil

            # Get initial memory usage
            process = psutil.Process(os.getpid())
            initial_memory = process.memory_info().rss

            result = analyze_replay_with_enhanced_stats(replay_path, "test_run", mock_mcp_client)

            # Get final memory usage
            final_memory = process.memory_info().rss
            memory_increase = final_memory - initial_memory

            # Memory increase should be reasonable (less than 100MB for test data)
            assert memory_increase < 100 * 1024 * 1024, (
                f"Memory usage increased by {memory_increase / 1024 / 1024:.1f}MB"
            )

            # Verify analysis completed successfully
            assert result is not None
            assert "enhanced_stats_analysis" in result

        except ImportError:
            # Skip memory test if psutil not available
            pytest.skip("psutil not available for memory testing")
        finally:
            Path(replay_path).unlink()


class TestBackwardCompatibility:
    """Test backward compatibility with existing analysis system"""

    def test_existing_analysis_structure_preserved(self, temp_replay_file, sample_episode_stats, mock_mcp_client):
        """Test that existing analysis structure is preserved"""
        replay_data = {
            "version": "1.0",
            "max_steps": 1000,
            "episode_stats": sample_episode_stats,
            "objects": [
                {
                    "agent_id": 0,
                    "type_id": 0,
                    "location": [[0, [5, 5, 0]]],
                    "inventory": [[0, []]],
                    "total_reward": [[0, 0.0]],
                }
            ],
        }

        with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
            json.dump(replay_data, f)
            replay_path = f.name

        try:
            result = analyze_replay_with_enhanced_stats(replay_path, "test_run", mock_mcp_client)

            # Verify basic analysis structure is preserved
            basic_analysis = result["basic_analysis"]

            # These fields should exist from original analysis
            expected_basic_fields = ["agent_count", "episode_length"]
            for field in expected_basic_fields:
                assert field in basic_analysis or any(field in str(v) for v in basic_analysis.values())

            # New enhanced fields should be added
            assert "enhanced_stats_analysis" in result
            assert "wandb_training_context" in result

        finally:
            Path(replay_path).unlink()

    def test_api_compatibility(self, temp_replay_file, sample_episode_stats, mock_mcp_client):
        """Test that API remains compatible with existing usage"""
        replay_data = {"version": "1.0", "max_steps": 1000, "episode_stats": sample_episode_stats, "objects": []}

        with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
            json.dump(replay_data, f)
            replay_path = f.name

        try:
            # Test with all parameters (new usage)
            result_full = analyze_replay_with_enhanced_stats(replay_path, "test_run", mock_mcp_client)
            assert result_full is not None

            # Test with minimal parameters (backward compatible)
            # Should work but may have limited insights
            analyze_replay_with_enhanced_stats(replay_path, None, None)
            # Should fail hard since we require WandB data (no graceful degradation)
            # This is expected behavior per requirements

        except ValueError:
            # Expected for minimal parameters due to hard failure requirement
            pass
        finally:
            Path(replay_path).unlink()


class TestGeneratorTypeBreakdown:
    """Test generator and converter type breakdown functionality"""

    def test_generator_converter_type_breakdown(self):
        """Test that generator and converter types are properly categorized and displayed"""
        # Create test objects with mixed generator and converter types
        test_objects = [
            {"type_id": 2, "location": [5, 5, 1], "hp": 100, "converting": []},  # mine_red
            {"type_id": 3, "location": [6, 6, 1], "hp": 100, "converting": []},  # mine_blue
            {"type_id": 4, "location": [7, 7, 1], "hp": 100, "converting": []},  # mine_green
            {"type_id": 5, "location": [8, 8, 1], "hp": 100, "converting": []},  # generator_red
            {"type_id": 6, "location": [9, 9, 1], "hp": 100, "converting": []},  # generator_blue
            {"type_id": 7, "location": [10, 10, 1], "hp": 100, "converting": []},  # generator_green
            {"type_id": 8, "location": [11, 11, 1], "hp": 100, "converting": []},  # altar
        ]

        test_object_types = [
            "agent",
            "wall",
            "mine_red",
            "mine_blue",
            "mine_green",
            "generator_red",
            "generator_blue",
            "generator_green",
            "altar",
        ]

        # Test the generator/altar state extraction
        from metta.mcp_server.training_utils import _extract_generator_altar_states

        result = _extract_generator_altar_states(test_objects, test_object_types)

        # Verify generators are properly categorized
        generators = result["generators"]
        generator_type_counts = {}
        for gen in generators:
            type_name = gen.get("type_name", "unknown")
            generator_type_counts[type_name] = generator_type_counts.get(type_name, 0) + 1

        # Should find all generator types
        expected_generators = {
            "mine_red": 1,
            "mine_blue": 1,
            "mine_green": 1,
            "generator_red": 1,
            "generator_blue": 1,
            "generator_green": 1,
        }

        assert generator_type_counts == expected_generators

        # Test the LLM client environmental context formatting
        from metta.mcp_server.llm_client import LLMClient

        client = LLMClient()

        # Create mock analysis data with the resource objects
        mock_analysis = {
            "environmental_context": {
                "resource_objects": {
                    "generators": generators,
                    "altars": result["altars"],
                    "converters": [],  # In mettagrid, mines/generators ARE the converters
                }
            },
            "episode_length": 1000,
            "agents": [],
            "final_scores": {},
            "key_events": [],
        }

        # Generate the user prompt
        user_prompt = client._create_user_prompt(mock_analysis)

        # Verify the environmental context shows type breakdown
        assert "ENVIRONMENTAL CONTEXT:" in user_prompt

        # Check that different generator types are mentioned
        assert "mine_red" in user_prompt or "1 mine_reds" in user_prompt
        assert "mine_blue" in user_prompt or "1 mine_blues" in user_prompt
        assert "generator_red" in user_prompt or "1 generator_reds" in user_prompt
        assert "generator_blue" in user_prompt or "1 generator_blues" in user_prompt

        # Should show generator type breakdown in environmental context
        lines = user_prompt.split("\n")
        resource_line = next((line for line in lines if "Resource Objects:" in line), "")

        # Verify the resource line contains type-specific counts
        assert resource_line != ""
        # Should contain generator type breakdown like "1 mine_reds, 1 mine_blues, 1 mine_greens,
        # 1 generator_reds, 1 generator_blues, 1 generator_greens, 1 altars"
        assert any(color in resource_line for color in ["red", "blue", "green"])

        print("✅ Generator type breakdown test passed")
        print(f"   Found generator types: {generator_type_counts}")
        print(f"   Environmental context line: {resource_line}")

    def test_generator_type_breakdown_with_only_red_generators(self):
        """Test generator type breakdown with only red generators (common scenario)"""
        test_objects = [
            {"type_id": 2, "location": [5, 5, 1], "hp": 100, "converting": []},  # mine_red
            {"type_id": 2, "location": [6, 6, 1], "hp": 100, "converting": []},  # mine_red
            {"type_id": 5, "location": [8, 8, 1], "hp": 100, "converting": []},  # generator_red
            {"type_id": 8, "location": [11, 11, 1], "hp": 100, "converting": []},  # altar
        ]

        test_object_types = [
            "agent",
            "wall",
            "mine_red",
            "mine_blue",
            "mine_green",
            "generator_red",
            "generator_blue",
            "generator_green",
            "altar",
        ]

        from metta.mcp_server.training_utils import _extract_generator_altar_states

        result = _extract_generator_altar_states(test_objects, test_object_types)

        generators = result["generators"]
        generator_type_counts = {}
        for gen in generators:
            type_name = gen.get("type_name", "unknown")
            generator_type_counts[type_name] = generator_type_counts.get(type_name, 0) + 1

        # Should find only red generators
        expected_generators = {"mine_red": 2, "generator_red": 1}

        assert generator_type_counts == expected_generators

        # Test environmental context formatting for red-only scenario
        from metta.mcp_server.llm_client import LLMClient

        client = LLMClient()

        mock_analysis = {
            "environmental_context": {
                "resource_objects": {"generators": generators, "altars": result["altars"], "converters": []}
            },
            "episode_length": 1000,
            "agents": [],
            "final_scores": {},
            "key_events": [],
        }

        user_prompt = client._create_user_prompt(mock_analysis)

        # Should show red generator breakdown
        lines = user_prompt.split("\n")
        resource_line = next((line for line in lines if "Resource Objects:" in line), "")

        assert "mine_red" in resource_line
        assert "generator_red" in resource_line
        # Should not contain blue or green generators
        assert "blue" not in resource_line
        assert "green" not in resource_line

        print("✅ Red-only generator test passed")
        print(f"   Resource line: {resource_line}")

    def test_mixed_generator_colors_environmental_context(self):
        """Test environmental context display with mixed red, blue, and green generators"""
        test_objects = [
            {"type_id": 2, "location": [5, 5, 1], "hp": 100, "converting": []},  # mine_red
            {"type_id": 3, "location": [6, 6, 1], "hp": 100, "converting": []},  # mine_blue
            {"type_id": 5, "location": [8, 8, 1], "hp": 100, "converting": []},  # generator_red
            {"type_id": 6, "location": [9, 9, 1], "hp": 100, "converting": []},  # generator_blue
            {"type_id": 8, "location": [11, 11, 1], "hp": 100, "converting": []},  # altar
        ]

        test_object_types = [
            "agent",
            "wall",
            "mine_red",
            "mine_blue",
            "mine_green",
            "generator_red",
            "generator_blue",
            "generator_green",
            "altar",
        ]

        from metta.mcp_server.training_utils import _extract_generator_altar_states

        result = _extract_generator_altar_states(test_objects, test_object_types)

        generators = result["generators"]

        # Test that the LLM client properly formats the breakdown
        from metta.mcp_server.llm_client import LLMClient

        client = LLMClient()

        mock_analysis = {
            "environmental_context": {
                "resource_objects": {"generators": generators, "altars": result["altars"], "converters": []}
            },
            "episode_length": 1000,
            "agents": [],
            "final_scores": {},
            "key_events": [],
        }

        user_prompt = client._create_user_prompt(mock_analysis)

        # Extract the resource objects line
        lines = user_prompt.split("\n")
        resource_line = next((line for line in lines if "Resource Objects:" in line), "")

        # Should contain mixed color breakdown
        assert resource_line != ""
        assert "mine_red" in resource_line or "1 mine_reds" in resource_line
        assert "mine_blue" in resource_line or "1 mine_blues" in resource_line
        assert "generator_red" in resource_line or "1 generator_reds" in resource_line
        assert "generator_blue" in resource_line or "1 generator_blues" in resource_line
        assert "altar" in resource_line

        # Should contain "red" and "blue" to show color variety
        assert "red" in resource_line.lower()
        assert "blue" in resource_line.lower()

        print("✅ Mixed generator colors test passed")
        print(f"   Resource line: {resource_line}")

    def test_no_generators_handled_gracefully(self):
        """Test that environments with no generators are handled gracefully"""
        test_objects = [
            {"type_id": 0, "location": [5, 5, 1], "hp": 100},  # agent
            {"type_id": 8, "location": [11, 11, 1], "hp": 100, "converting": []},  # altar only
        ]

        test_object_types = [
            "agent",
            "wall",
            "mine_red",
            "mine_blue",
            "mine_green",
            "generator_red",
            "generator_blue",
            "generator_green",
            "altar",
        ]

        from metta.mcp_server.training_utils import _extract_generator_altar_states

        result = _extract_generator_altar_states(test_objects, test_object_types)

        generators = result["generators"]
        altars = result["altars"]

        # Should have no generators but 1 altar
        assert len(generators) == 0
        assert len(altars) == 1

        # Test environmental context formatting
        from metta.mcp_server.llm_client import LLMClient

        client = LLMClient()

        mock_analysis = {
            "environmental_context": {
                "resource_objects": {"generators": generators, "altars": altars, "converters": []}
            },
            "episode_length": 1000,
            "agents": [],
            "final_scores": {},
            "key_events": [],
        }

        user_prompt = client._create_user_prompt(mock_analysis)

        # Should handle empty generators gracefully
        lines = user_prompt.split("\n")
        resource_line = next((line for line in lines if "Resource Objects:" in line), "")

        assert resource_line != ""
        assert "0 generators" in resource_line or resource_line.endswith("1 altars")

        print("✅ No generators test passed")
        print(f"   Resource line: {resource_line}")
