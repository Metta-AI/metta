"""Tests for the run_tool preprocessor that handles short recipe syntax."""


from metta.common.tool.run_tool import preprocess_recipe_path


class TestPreprocessRecipePath:
    """Test the preprocess_recipe_path function."""

    def test_known_tools_convert_correctly(self):
        """Test that known tool names are converted to their specific recipe functions."""
        # Test each known tool
        assert preprocess_recipe_path("train arena") == "experiments.recipes.arena.train_recipe"
        assert preprocess_recipe_path("evaluate navigation") == "experiments.recipes.navigation.evaluate_recipe"
        assert preprocess_recipe_path("analyze minimal") == "experiments.recipes.minimal.analyze_recipe"
        assert (
            preprocess_recipe_path("play arena_basic_easy_shaped")
            == "experiments.recipes.arena_basic_easy_shaped.play_recipe"
        )
        assert preprocess_recipe_path("replay custom_env") == "experiments.recipes.custom_env.replay_recipe"
        assert preprocess_recipe_path("sim my_recipe") == "experiments.recipes.my_recipe.sim_recipe"

    def test_unknown_tools_fallback_to_mettagrid_recipe(self):
        """Test that unknown tool names fall back to mettagrid_recipe."""
        # Test various unknown tool names
        assert preprocess_recipe_path("custom_tool arena") == "experiments.recipes.arena.mettagrid_recipe"
        assert preprocess_recipe_path("mycustomtool navigation") == "experiments.recipes.navigation.mettagrid_recipe"
        assert preprocess_recipe_path("special minimal") == "experiments.recipes.minimal.mettagrid_recipe"
        assert preprocess_recipe_path("unknown_command my_env") == "experiments.recipes.my_env.mettagrid_recipe"

    def test_full_paths_unchanged(self):
        """Test that full module paths are returned unchanged."""
        # Test full paths that shouldn't be modified
        assert (
            preprocess_recipe_path("experiments.recipes.arena.train_recipe") == "experiments.recipes.arena.train_recipe"
        )
        assert preprocess_recipe_path("my.custom.module.function") == "my.custom.module.function"
        assert preprocess_recipe_path("some_module.some_function") == "some_module.some_function"

    def test_single_word_unchanged(self):
        """Test that single words are returned unchanged."""
        # Single words shouldn't be processed
        assert preprocess_recipe_path("train") == "train"
        assert preprocess_recipe_path("arena") == "arena"
        assert preprocess_recipe_path("something") == "something"

    def test_multiple_words_unchanged(self):
        """Test that inputs with more than 2 words are returned unchanged."""
        # More than 2 parts shouldn't be processed
        assert preprocess_recipe_path("train arena extra") == "train arena extra"
        assert preprocess_recipe_path("too many words here") == "too many words here"

    def test_recipe_names_with_underscores(self):
        """Test that recipe names with underscores work correctly."""
        # Recipe names can have underscores
        assert preprocess_recipe_path("train arena_basic_easy") == "experiments.recipes.arena_basic_easy.train_recipe"
        assert preprocess_recipe_path("play my_custom_recipe") == "experiments.recipes.my_custom_recipe.play_recipe"
        assert (
            preprocess_recipe_path("custom arena_with_underscores")
            == "experiments.recipes.arena_with_underscores.mettagrid_recipe"
        )

    def test_edge_cases(self):
        """Test edge cases and boundary conditions."""
        # Empty string
        assert preprocess_recipe_path("") == ""

        # Whitespace handling
        assert preprocess_recipe_path("  train  arena  ") == "experiments.recipes.arena.train_recipe"
        assert preprocess_recipe_path("train\tarena") == "experiments.recipes.arena.train_recipe"

        # Case sensitivity (assuming the function is case-sensitive)
        assert (
            preprocess_recipe_path("Train arena") == "experiments.recipes.arena.mettagrid_recipe"
        )  # Capital T, not recognized
        assert (
            preprocess_recipe_path("TRAIN ARENA") == "experiments.recipes.ARENA.mettagrid_recipe"
        )  # All caps, not recognized

    def test_all_known_tools_covered(self):
        """Ensure all known tool mappings work correctly."""
        known_tools = ["train", "evaluate", "analyze", "play", "replay", "sim"]

        for tool in known_tools:
            result = preprocess_recipe_path(f"{tool} test_recipe")
            assert result == f"experiments.recipes.test_recipe.{tool}_recipe"
            assert "mettagrid_recipe" not in result  # Should not fall back

    def test_fallback_preserves_recipe_name(self):
        """Test that fallback preserves the exact recipe name provided."""
        # The recipe name should be preserved exactly as given
        assert (
            preprocess_recipe_path("custom CamelCaseRecipe") == "experiments.recipes.CamelCaseRecipe.mettagrid_recipe"
        )
        assert (
            preprocess_recipe_path("tool recipe-with-dash") == "experiments.recipes.recipe-with-dash.mettagrid_recipe"
        )
        assert preprocess_recipe_path("cmd recipe.with.dots") == "experiments.recipes.recipe.with.dots.mettagrid_recipe"
