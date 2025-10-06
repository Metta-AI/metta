"""Tests for recipe discovery without __init__.py files."""

from metta.common.tool.recipe_registry import recipe_registry


def test_recipe_discovery_without_init():
    """Test that recipes are discovered even in subdirectories without __init__.py."""
    recipe_registry.clear()  # Start fresh
    recipe_registry.discover_all()

    recipes = {r.module_name for r in recipe_registry.get_all()}

    # Should find recipes in subdirectories without __init__.py
    # Check for a known recipe in a subdirectory
    assert any("in_context_learning" in name for name in recipes), (
        f"Should discover recipes in subdirectories. Found: {sorted(recipes)}"
    )

    # Should find top-level recipes
    assert "experiments.recipes.arena" in recipes, f"Should find top-level recipes. Found: {sorted(recipes)}"

    # Should skip __init__.py files
    assert "experiments.recipes.__init__" not in recipes, "Should skip __init__.py files"

    # Should skip private modules
    assert not any(name.split(".")[-1].startswith("_") for name in recipes), (
        f"Should skip private modules. Found: {sorted(recipes)}"
    )


def test_recipe_registry_get_normalizes_paths():
    """Test that RecipeRegistry.get() handles both short and full paths."""

    recipe_registry.clear()
    recipe_registry.discover_all()

    # Should work with full path
    recipe_full = recipe_registry.get("experiments.recipes.arena")
    assert recipe_full is not None, "Should find recipe with full path"

    # Should work with short path
    recipe_short = recipe_registry.get("arena")
    assert recipe_short is not None, "Should find recipe with short path"

    # Should be the same recipe
    assert recipe_full.module_name == recipe_short.module_name

    # Should work with subdirectory (short form)
    recipe_sub = recipe_registry.get("in_context_learning.in_context_learning")
    if recipe_sub:  # Only test if this recipe exists
        assert recipe_sub.module_name.startswith("experiments.recipes.")


def test_recipe_short_name():
    """Test Recipe.short_name property."""

    recipe_registry.clear()
    recipe_registry.discover_all()

    recipe = recipe_registry.get("arena")
    assert recipe is not None

    assert recipe.module_name == "experiments.recipes.arena"
    assert recipe.short_name == "arena"

    # Test with subdirectory
    recipe_sub = recipe_registry.get("in_context_learning.in_context_learning")
    if recipe_sub:
        assert recipe_sub.short_name == "in_context_learning.in_context_learning"
