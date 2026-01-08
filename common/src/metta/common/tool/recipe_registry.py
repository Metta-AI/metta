"""Recipe registry for discovering and caching recipe modules."""

from __future__ import annotations

import importlib
import importlib.util
import pkgutil

from metta.common.tool.recipe import Recipe


class RecipeRegistry:
    """Singleton registry for discovered recipes.

    Access recipes via `.path_to_recipe` attribute.
    """

    _instance: RecipeRegistry | None = None
    path_to_recipe: dict[str, Recipe]  # module_path -> Recipe
    _discovered: bool = False

    def __new__(cls) -> RecipeRegistry:
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            cls._instance.path_to_recipe = {}
            cls._instance._discovered = False
        return cls._instance

    def _ensure_discovered(self) -> None:
        """Lazily discover all recipes on first access."""
        if not self._discovered:
            # Discover from both prod and experiment locations
            self.discover_all("recipes.prod")
            self.discover_all("recipes.experiment")
            self._discovered = True

    def get(self, module_path: str) -> Recipe | None:
        """Get a recipe by module path (tries both short and full paths)."""
        self._ensure_discovered()

        # Try exact match first
        if module_path in self.path_to_recipe:
            return self.path_to_recipe[module_path]

        # Try with prefix if it's a short name
        if not module_path.startswith("recipes."):
            # Try prod first, then experiment
            for prefix in ["recipes.prod", "recipes.experiment"]:
                full_path = f"{prefix}.{module_path}"
                if full_path in self.path_to_recipe:
                    return self.path_to_recipe[full_path]

        # Try to load directly as a fallback for recipes outside recipes package
        # This supports test fixtures and external recipe packages
        recipe = Recipe.load(module_path)
        if recipe:
            # Check if it has tool makers (valid recipe)
            if recipe.get_explicit_tool_makers():
                # Cache it for future lookups
                self.path_to_recipe[module_path] = recipe
                return recipe

        return None

    def get_all(self) -> list[Recipe]:
        """Get all discovered recipes."""
        self._ensure_discovered()
        return list(self.path_to_recipe.values())

    def discover_all(self, base_package: str = "recipes.prod") -> None:
        """Discover all recipe modules under a base package and add to registry.

        Uses pkgutil.walk_packages() to recursively walk all subpackages.
        Requires __init__.py files for proper package structure.

        Args:
            base_package: Base package to search for recipes (e.g., recipes.prod, recipes.experiment)
        """
        if importlib.util.find_spec(base_package) is None:
            return None
        base_module = importlib.import_module(base_package)

        # Get the package path
        if not hasattr(base_module, "__path__"):
            return

        # Walk packages recursively
        for _importer, modname, ispkg in pkgutil.walk_packages(path=base_module.__path__, prefix=f"{base_package}."):
            # Skip private modules
            if any(part.startswith("_") for part in modname.split(".")):
                continue

            # Try to load as recipe (skip packages, only load modules)
            if not ispkg:
                recipe = Recipe.load(modname)
                if recipe:
                    # Only include if it has tool makers
                    if recipe.get_explicit_tool_makers():
                        self.path_to_recipe[modname] = recipe

    def clear(self) -> None:
        """Clear the registry (mainly for testing)."""
        self.path_to_recipe.clear()
        self._discovered = False


# Global singleton - access directly
recipe_registry = RecipeRegistry()
