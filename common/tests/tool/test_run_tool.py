import os
import re
import subprocess
import sys

import pytest
from metta.common.tool.run_tool import get_tool_type_mapping, preprocess_recipe_path


def find_with_whitespace(target: str, text: str) -> bool:
    """
    Find a target string in text, allowing for arbitrary whitespace between words.
    """
    # Split the search string into words and escape regex special characters
    words = [re.escape(word) for word in target.split()]
    # Join with \s+ to match any whitespace between words
    pattern = r"\s+".join(words)
    return bool(re.search(pattern, text))


@pytest.fixture
def with_extra_imports_root(monkeypatch):
    extra_imports_root = os.path.join(os.path.dirname(__file__), "fixtures/extra-import-root")
    monkeypatch.setenv("PYTHONPATH", extra_imports_root)


def run_tool(*args: str) -> subprocess.CompletedProcess[str]:
    """Helper to invoke the runner with text I/O, capturing both streams."""
    return subprocess.run(
        [sys.executable, "-m", "metta.common.tool.run_tool", *args],
        text=True,
        capture_output=True,
        check=False,
    )


def test_basic(with_extra_imports_root):
    # Behavior: the tool runs and prints from invoke()
    result = subprocess.check_output(
        [sys.executable, "-m", "metta.common.tool.run_tool", "mypackage.tools.TestTool"], text=True
    )
    assert find_with_whitespace("TestTool invoked", result)


def test_unknown_tool(with_extra_imports_root):
    # Behavior: unknown symbol yields non-zero with helpful error content
    result = run_tool("mypackage.tools.NoSuchTool")
    assert result.returncode > 0
    combined_output = result.stderr + result.stdout
    assert find_with_whitespace("has no", combined_output)
    assert "attribute" in combined_output
    assert "NoSuchTool" in combined_output


def test_unknown_module(with_extra_imports_root):
    # Behavior: unknown module yields non-zero with ImportError details
    result = run_tool("mypackage.no_such_tools.TestTool")
    assert result.returncode > 0
    combined_output = result.stderr + result.stdout
    assert find_with_whitespace("No module named", combined_output)
    assert "mypackage.no_such_tools" in combined_output


def test_required_field_supported_for_tool_class(with_extra_imports_root):
    """
    Behavior: required Pydantic field on a Tool subclass is validated at construction.
    Edge case from review: `x=123` must be accepted and printed.
    ""
    result = run_tool("mypackage.tools.RequiredFieldTool", "x=123")
    assert result.returncode == 0
    combined_output = result.stdout + result.stderr
    assert "123" in combined_output


def test_dotted_overrides_are_nested_and_validated(with_extra_imports_root):
    """Test that dotted keys are interpreted as nested config and validated at construction time."""
    result = run_tool(
        "mypackage.tools.SimpleTestTool",
        "value=custom",
        "nested.field=updated",
        "nested.another_field=999",
    )
    assert result.returncode == 0
    out = result.stdout + result.stderr
    # Printed by SimpleTestTool.invoke
    assert find_with_whitespace("Tool value: custom", out)
    assert find_with_whitespace("Tool nested.field: updated", out)
    assert find_with_whitespace("Tool nested.another_field: 999", out)


def test_factory_function_params_and_invoke_args(with_extra_imports_root):
    """
    Behavior: factory function parameters are bound using type info and passed to invoke() as strings.
    SimpleTestTool prints the 'Args' dict it receives in invoke(); assert values are stringified.
    """
    result = run_tool(
        "mypackage.tools.make_test_tool",
        "run=my_test",
        "count=99",
        "value=override",
    )
    assert result.returncode == 0
    out = result.stdout + result.stderr
    # SimpleTestTool.invoke prints "Args: {...}" and its own fields
    # Runner passes only factory-function args (not overrides) to invoke(), as strings
    assert find_with_whitespace("Args: {'run': 'my_test', 'count': '99'}", out)
    assert find_with_whitespace("Tool value: override", out)


class TestPreprocessRecipePath:
    """Test the preprocess_recipe_path function."""

    def test_known_tools_convert_correctly(self):
        """Test that known tool names are converted to their specific recipe functions."""
        # Test each known tool with expected tool type
        tool_types = get_tool_type_mapping()

        path, tool_type = preprocess_recipe_path("train arena")
        assert path == "experiments.recipes.arena.train"
        assert tool_type == tool_types["train"]

        path, tool_type = preprocess_recipe_path("sim navigation")
        assert path == "experiments.recipes.navigation.sim"
        assert tool_type == tool_types["sim"]

        path, tool_type = preprocess_recipe_path("analyze minimal")
        assert path == "experiments.recipes.minimal.analyze"
        assert tool_type == tool_types["analyze"]

        path, tool_type = preprocess_recipe_path("play arena_basic_easy_shaped")
        assert path == "experiments.recipes.arena_basic_easy_shaped.play"
        assert tool_type == tool_types["play"]

        path, tool_type = preprocess_recipe_path("replay custom_env")
        assert path == "experiments.recipes.custom_env.replay"
        assert tool_type == tool_types["replay"]

        path, tool_type = preprocess_recipe_path("sim my_recipe")
        assert path == "experiments.recipes.my_recipe.sim"
        assert tool_type == tool_types["sim"]

    def test_unknown_tools_unchanged(self):
        """Test that unknown tool names are returned unchanged (not part of short syntax)."""
        # Unknown tool names are not processed by the preprocessor - no expected type
        assert preprocess_recipe_path("custom_tool arena") == ("custom_tool arena", None)
        assert preprocess_recipe_path("mycustomtool navigation") == ("mycustomtool navigation", None)
        assert preprocess_recipe_path("special minimal") == ("special minimal", None)
        assert preprocess_recipe_path("unknown_command my_env") == ("unknown_command my_env", None)

    def test_full_paths_unchanged(self):
        """Test that full module paths are returned unchanged."""
        # Test full paths that shouldn't be modified - no expected type
        path, tool_type = preprocess_recipe_path("experiments.recipes.arena.train")
        assert path == "experiments.recipes.arena.train"
        assert tool_type is None

        path, tool_type = preprocess_recipe_path("my.custom.module.function")
        assert path == "my.custom.module.function"
        assert tool_type is None

        path, tool_type = preprocess_recipe_path("some_module.some_function")
        assert path == "some_module.some_function"
        assert tool_type is None

    def test_single_word_unchanged(self):
        """Test that single words are returned unchanged."""
        # Single words shouldn't be processed - no expected type
        assert preprocess_recipe_path("train") == ("train", None)
        assert preprocess_recipe_path("arena") == ("arena", None)
        assert preprocess_recipe_path("something") == ("something", None)

    def test_multiple_words_unchanged(self):
        """Test that inputs with more than 2 words are returned unchanged."""
        # More than 2 parts shouldn't be processed - no expected type
        assert preprocess_recipe_path("train arena extra") == ("train arena extra", None)
        assert preprocess_recipe_path("too many words here") == ("too many words here", None)

    def test_recipe_names_with_underscores(self):
        """Test that recipe names with underscores work correctly."""
        tool_types = get_tool_type_mapping()
        # Recipe names can have underscores
        path, tool_type = preprocess_recipe_path("train arena_basic_easy")
        assert path == "experiments.recipes.arena_basic_easy.train"
        assert tool_type == tool_types["train"]

        path, tool_type = preprocess_recipe_path("play my_custom_recipe")
        assert path == "experiments.recipes.my_custom_recipe.play"
        assert tool_type == tool_types["play"]

        # Unknown tool names are not processed
        assert preprocess_recipe_path("custom arena_with_underscores") == ("custom arena_with_underscores", None)

    def test_dotted_recipe_names(self):
        """Test that dotted recipe names are preserved without appending _recipe when they match tool pattern."""
        tool_types = get_tool_type_mapping()
        # When the function name matches the tool pattern (tool_*), preserve it
        path, tool_type = preprocess_recipe_path("replay scratchpad.ci.replay_null")
        assert path == "experiments.recipes.scratchpad.ci.replay_null"
        assert tool_type == tool_types["replay"]

        path, tool_type = preprocess_recipe_path("play scratchpad.ci.play_null")
        assert path == "experiments.recipes.scratchpad.ci.play_null"
        assert tool_type == tool_types["play"]

        # When the function name doesn't match the tool pattern, append the tool name
        path, tool_type = preprocess_recipe_path("train module.submodule.function")
        assert path == "experiments.recipes.module.submodule.function.train"
        assert tool_type == tool_types["train"]

    def test_fully_qualified_with_function_preserved(self):
        """Full path with experiments.recipes.* and tool_* function should be preserved without double prefix."""
        tool_types = get_tool_type_mapping()
        path, tool_type = preprocess_recipe_path("train experiments.recipes.arena.train_special")
        assert path == "experiments.recipes.arena.train_special"
        assert tool_type == tool_types["train"]

        # Fully-qualified without function name appends the verb
        path, tool_type = preprocess_recipe_path("train experiments.recipes.arena")
        assert path == "experiments.recipes.arena.train"
        assert tool_type == tool_types["train"]

    def test_subfolder_recipes(self):
        """Test that subfolder recipes work correctly."""
        tool_types = get_tool_type_mapping()
        # Subfolder without function name gets tool name appended
        path, tool_type = preprocess_recipe_path("train in_context_learning.ordered_chains")
        assert path == "experiments.recipes.in_context_learning.ordered_chains.train"
        assert tool_type == tool_types["train"]

        path, tool_type = preprocess_recipe_path("play navigation_sequence.complex")
        assert path == "experiments.recipes.navigation_sequence.complex.play"
        assert tool_type == tool_types["play"]

        # Multiple levels of nesting - note: no 'evaluate' tool, should use 'sim'
        path, tool_type = preprocess_recipe_path("sim deep.nested.module")
        assert path == "experiments.recipes.deep.nested.module.sim"
        assert tool_type == tool_types["sim"]

    def test_edge_cases(self):
        """Test edge cases and boundary conditions."""
        tool_types = get_tool_type_mapping()
        # Empty string
        assert preprocess_recipe_path("") == ("", None)

        # Whitespace handling
        path, tool_type = preprocess_recipe_path("  train  arena  ")
        assert path == "experiments.recipes.arena.train"
        assert tool_type == tool_types["train"]

        path, tool_type = preprocess_recipe_path("train\tarena")
        assert path == "experiments.recipes.arena.train"
        assert tool_type == tool_types["train"]

        # Case sensitivity (assuming the function is case-sensitive)
        # Unknown tools are not processed
        assert preprocess_recipe_path("Train arena") == ("Train arena", None)  # Capital T, not recognized
        assert preprocess_recipe_path("TRAIN ARENA") == ("TRAIN ARENA", None)  # All caps, not recognized

    def test_all_known_tools_covered(self):
        """Ensure all known tool mappings work correctly."""
        known_tools = ["train", "sim", "analyze", "play", "replay"]  # No 'evaluate'
        tool_types = get_tool_type_mapping()

        for tool in known_tools:
            path, tool_type = preprocess_recipe_path(f"{tool} test_recipe")
            assert path == f"experiments.recipes.test_recipe.{tool}"
            assert tool_type == tool_types[tool]
            assert "mettagrid" not in path  # Should not fall back

    def test_fallback_behavior_documented(self):
        """Document that fallback to mettagrid_recipe happens at runtime, not in preprocessor."""
        # The preprocessor only handles known tools
        # The fallback to mettagrid_recipe happens at runtime when loading the module
        # For known tools, the preprocessor generates the expected path
        tool_types = get_tool_type_mapping()
        path, tool_type = preprocess_recipe_path("train simple")
        assert path == "experiments.recipes.simple.train"
        assert tool_type == tool_types["train"]
        # Even if train doesn't exist, the preprocessor still returns this path
        # The runtime loader will then fall back to mettagrid if train is missing

    def test_single_path_preserved(self):
        """Test that single-path inputs are preserved as-is."""
        # Single paths are not modified by the preprocessor - no expected type
        # The runtime loader will try them as-is, then with experiments.recipes prefix
        path, tool_type = preprocess_recipe_path("in_context_learning.ordered_chains.train")
        assert path == "in_context_learning.ordered_chains.train"
        assert tool_type is None

        path, tool_type = preprocess_recipe_path("some.module.function")
        assert path == "some.module.function"
        assert tool_type is None

        path, tool_type = preprocess_recipe_path("experiments.recipes.arena.train")
        assert path == "experiments.recipes.arena.train"
        assert tool_type is None

    def test_direct_function_path(self):
        """Test direct function path syntax (arena.train_shaped) for backwards compatibility."""
        # Direct function paths should be preserved as-is, with no expected tool type
        path, tool_type = preprocess_recipe_path("arena.train_shaped")
        assert path == "arena.train_shaped"
        assert tool_type is None

        path, tool_type = preprocess_recipe_path("navigation.evaluate_special")
        assert path == "navigation.evaluate_special"
        assert tool_type is None

        path, tool_type = preprocess_recipe_path("minimal.sim_custom")
        assert path == "minimal.sim_custom"
        assert tool_type is None

        # These will get experiments.recipes prefix added by the loader if not found
        # The preprocessor doesn't modify them
