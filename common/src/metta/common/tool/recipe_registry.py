"""Recipe registry for discovering and caching recipe modules."""

from __future__ import annotations

import importlib
from pathlib import Path

from metta.common.tool.recipe import Recipe


class RecipeRegistry:
    """Singleton registry for discovered recipes."""

    _instance: RecipeRegistry | None = None
    _recipes: dict[str, Recipe]  # module_name -> Recipe
    _discovered: bool = False

    def __new__(cls) -> RecipeRegistry:
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            cls._instance._recipes = {}
            cls._instance._discovered = False
        return cls._instance

    def _ensure_discovered(self) -> None:
        """Lazily discover all recipes on first access."""
        if not self._discovered:
            self.discover_all()
            self._discovered = True

    def get(self, module_path: str) -> Recipe | None:
        """Get a recipe by module path (tries both short and full paths)."""
        self._ensure_discovered()

        # Try exact match first
        if module_path in self._recipes:
            return self._recipes[module_path]

        # Try with prefix if it's a short name
        if not module_path.startswith("experiments.recipes."):
            full_path = f"experiments.recipes.{module_path}"
            if full_path in self._recipes:
                return self._recipes[full_path]

        return None

    def has(self, module_path: str) -> bool:
        """Check if a recipe exists at the given module path."""
        return self.get(module_path) is not None

    def get_all(self) -> list[Recipe]:
        """Get all discovered recipes."""
        self._ensure_discovered()
        return list(self._recipes.values())

    def discover_all(self, base_package: str = "experiments.recipes") -> None:
        """Discover all recipe modules under a base package and add to registry.

        Uses file-system walking to find all .py files, not just packages with __init__.py.

        Args:
            base_package: Base package to search for recipes (default: experiments.recipes)
        """
        try:
            base_module = importlib.import_module(base_package)
        except ImportError:
            return

        # Get the package path
        if not hasattr(base_module, "__path__"):
            return

        # Walk filesystem to find all .py files (not just packages)
        for package_path in base_module.__path__:
            base_dir = Path(package_path)
            if not base_dir.exists():
                continue

            # Find all .py files recursively
            for py_file in base_dir.rglob("*.py"):
                # Skip __init__.py and private modules
                if py_file.name.startswith("_"):
                    continue

                # Convert file path to module name
                rel_path = py_file.relative_to(base_dir)
                module_parts = list(rel_path.parts[:-1]) + [rel_path.stem]
                modname = f"{base_package}.{'.'.join(module_parts)}"

                # Try to load as recipe
                recipe = Recipe.load(modname)
                if recipe:
                    # Only include if it has tools or configs
                    if recipe.get_explicit_tools() or recipe.get_configs() != (None, None):
                        self._recipes[modname] = recipe

    def clear(self) -> None:
        """Clear the registry (mainly for testing)."""
        self._recipes.clear()
        self._discovered = False


_recipe_registry = RecipeRegistry()


def get_recipe_registry() -> RecipeRegistry:
    """Get the global RecipeRegistry singleton instance."""
    return _recipe_registry
