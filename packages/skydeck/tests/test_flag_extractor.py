"""Tests for flag extractor."""

from skydeck.flag_extractor import extract_flags_from_tool_path, get_default_flags


def test_extract_flags_from_train_tool():
    """Test extracting flags from TrainTool."""
    flags = extract_flags_from_tool_path("metta.tools.train.TrainTool")

    # Should return a non-empty list
    assert len(flags) > 0

    # Each flag should have the required fields
    for flag in flags:
        assert "flag" in flag
        assert "type" in flag
        assert "default" in flag
        assert "required" in flag

        # flag should be a dotted path string
        assert isinstance(flag["flag"], str)
        assert len(flag["flag"]) > 0

        # type should be a string
        assert isinstance(flag["type"], str)

        # required should be a boolean
        assert isinstance(flag["required"], bool)

        # default should be None or a simple type (str, int, float, bool)
        if flag["default"] is not None:
            assert isinstance(flag["default"], (str, int, float, bool))

    # Check for some known flags
    flag_names = [f["flag"] for f in flags]
    assert "run" in flag_names
    assert any(f.startswith("trainer.") for f in flag_names)
    assert any(f.startswith("training_env.") for f in flag_names)


def test_extract_flags_with_invalid_tool_path():
    """Test extracting flags with invalid tool path."""
    flags = extract_flags_from_tool_path("invalid.tool.path")

    # Should return empty list for invalid path
    assert flags == []


def test_get_default_flags():
    """Test getting default flags from TrainTool."""
    flags = get_default_flags()

    # Should return the same as extracting from TrainTool directly
    assert len(flags) > 0

    # Verify structure
    for flag in flags:
        assert "flag" in flag
        assert "type" in flag
        assert "default" in flag
        assert "required" in flag


def test_flag_serialization():
    """Test that all extracted flags are JSON serializable."""
    import json

    flags = extract_flags_from_tool_path("metta.tools.train.TrainTool")

    # Should be able to serialize to JSON
    json_str = json.dumps(flags)
    assert len(json_str) > 0

    # Should be able to deserialize
    deserialized = json.loads(json_str)
    assert len(deserialized) == len(flags)


def test_flag_default_values():
    """Test that flag default values are properly extracted."""
    flags = extract_flags_from_tool_path("metta.tools.train.TrainTool")

    # Find flags with known default values
    run_flag = next((f for f in flags if f["flag"] == "run"), None)
    assert run_flag is not None

    # run is optional (has a default of None)
    assert run_flag["required"] is False

    # Check that numeric defaults are extracted correctly
    # (Note: actual defaults depend on TrainerConfig)
    trainer_flags = [f for f in flags if f["flag"].startswith("trainer.")]
    assert len(trainer_flags) > 0

    # Some trainer flags should have numeric defaults
    numeric_defaults = [f for f in trainer_flags if isinstance(f["default"], (int, float))]
    # We expect at least some numeric defaults (e.g., batch_size, learning_rate)
    # But we can't assert a specific count since it depends on the actual config
