"""
Eval Finder Widget for Jupyter Notebooks

An anywidget-based implementation for discovering and selecting evaluations.
Provides filtering by agent type, difficulty, category, and shows prerequisite relationships.
"""

import pathlib
from typing import Any, Callable, Dict, List

import anywidget
import traitlets

# FIXME: dev mode is actually broken for some reason. marybe it's marimo?
# FIXME: we need something like `dotenv` and `.env.local` files up in here.
_DEV = False
# _DEV = True

bundled_assets_dir = pathlib.Path(__file__).parent / "static"
src_dir = pathlib.Path(__file__).parent / "../src"

if _DEV:
    ESM = "http://localhost:5175/src/index.js?anywidget"
else:
    ESM = (bundled_assets_dir / "index.js").read_text()

CSS = (src_dir / "styles.css").read_text()


class EvalFinderWidget(anywidget.AnyWidget):
    """
    Interactive eval finder widget for discovering and selecting evaluations.

    Displays evaluations in a tree/list structure with filtering capabilities
    for agent compatibility, difficulty, and categories.
    """

    # AnyWidget requires _esm property for JavaScript code
    _esm = ESM
    _css = CSS
    name = traitlets.Unicode("EvalFinderWidget").tag(sync=True)

    # Widget traits (data that syncs between Python and JavaScript)
    eval_data = traitlets.Dict({}).tag(
        sync=True
    )  # Contains evaluations, tree structure, etc.
    selected_evals = traitlets.List([]).tag(sync=True)  # List of selected eval names

    # Filter settings
    category_filter = traitlets.List([]).tag(sync=True)  # Empty means all categories

    # UI state
    view_mode = traitlets.Unicode("category").tag(
        sync=True
    )  # "tree", "list", "category"
    search_term = traitlets.Unicode("").tag(sync=True)
    show_prerequisites = traitlets.Bool(True).tag(sync=True)

    # Event communication
    selection_changed = traitlets.Dict(allow_none=True, default_value=None).tag(
        sync=True
    )
    filter_changed = traitlets.Dict(allow_none=True, default_value=None).tag(sync=True)

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self._callbacks = {
            "selection_changed": [],
            "filter_changed": [],
        }

        # print("🔍 EvalFinderWidget initialized successfully!")

        # Set up observers
        self.observe(self._on_selection_changed, names="selection_changed")
        self.observe(self._on_filter_changed, names="filter_changed")

    def _on_selection_changed(self, change):
        """Handle eval selection events from JavaScript."""
        if change["new"]:
            for callback in self._callbacks["selection_changed"]:
                callback(change["new"])

    def _on_filter_changed(self, change):
        """Handle filter change events from JavaScript."""
        if change["new"]:
            for callback in self._callbacks["filter_changed"]:
                callback(change["new"])

    def on_selection_changed(self, callback: Callable[[Dict[str, Any]], None]):
        """Register a callback for when eval selection changes.

        Args:
            callback: Function that receives {'selected_evals': List[str], 'action': str}
        """
        self._callbacks["selection_changed"].append(callback)

    def on_filter_changed(self, callback: Callable[[Dict[str, Any]], None]):
        """Register a callback for when filters change.

        Args:
            callback: Function that receives filter change information
        """
        self._callbacks["filter_changed"].append(callback)

    def set_eval_data(
        self,
        evaluations: List[Dict[str, Any]],
        categories: List[Dict[str, Any]] | None = None,
    ):
        """Set the eval finder data.

        Args:
            evaluations: List of evaluation metadata dicts
            categories: Optional category structure for organized display.
                       If None, will build categories from all evaluations automatically.
        """
        # If no categories provided, build them from all evaluations
        if categories is None:
            categories = self._build_categories_from_evaluations(evaluations)

        self.eval_data = {
            "evaluations": evaluations,
            "categories": categories,
        }
        # print(f"📊 Eval data set with {len(evaluations)} evaluations")
        # print(f"📂 Category structure: {len(categories)} categories")
        # if categories:
        #     for category in categories:
        #         children_count = len(category.get("children", []))
        #         print(
        #             f"📂   - {category.get('name', 'unknown')}: {children_count} evaluations"
        #         )

    def _build_categories_from_evaluations(
        self, evaluations: List[Dict[str, Any]]
    ) -> List[Dict[str, Any]]:
        """Build category structure from evaluations automatically."""
        # Group evaluations by category
        categories_dict = {}
        for eval_data in evaluations:
            category = eval_data.get("category")
            if not category:
                continue

            if category not in categories_dict:
                categories_dict[category] = {
                    "name": category,
                    "category": category,
                    "children": [],
                }

            # Create eval node
            eval_node = {
                "name": eval_data["name"].split("/")[-1]
                if "/" in eval_data["name"]
                else eval_data["name"],
                "category": category,
                "eval_metadata": eval_data,
            }
            categories_dict[category]["children"].append(eval_node)

        # Convert to list and sort
        categories = list(categories_dict.values())
        categories.sort(key=lambda x: x["name"])
        return categories

    def set_category_filter(self, categories: List[str]):
        """Set the category filter.

        Args:
            categories: List of categories to include (empty means all)
        """
        self.category_filter = categories
        # filter_desc = ", ".join(categories) if categories else "all"
        # print(f"🗂️ Category filter set to: {filter_desc}")

    def set_view_mode(self, mode: str):
        """Set the display view mode.

        Args:
            mode: "tree", "list", or "category"
        """
        if mode in ["tree", "list", "category"]:
            self.view_mode = mode
            # print(f"👁️ View mode set to: {mode}")

    def get_selected_evals(self) -> List[str]:
        """Get the currently selected evaluation names.

        Returns:
            List of selected evaluation names
        """
        return self.selected_evals

    def select_evals(self, eval_names: List[str]):
        """Programmatically select evaluations.

        Args:
            eval_names: List of evaluation names to select
        """
        self.selected_evals = eval_names
        # print(f"✅ Selected {len(eval_names)} evaluations")

    def clear_selection(self):
        """Clear all selected evaluations."""
        self.selected_evals = []
        # print("🗑️ Selection cleared")

    def set_search_term(self, term: str):
        """Set the search filter term.

        Args:
            term: Search string to filter evaluations
        """
        self.search_term = term
        # print(f"🔍 Search term set to: '{term}'")


def create_eval_finder_widget(**kwargs) -> EvalFinderWidget:
    """Create and return a new eval finder widget.

    Args:
        **kwargs: Additional keyword arguments passed to EvalFinderWidget constructor

    Returns:
        EvalFinderWidget instance
    """
    return EvalFinderWidget(**kwargs)
