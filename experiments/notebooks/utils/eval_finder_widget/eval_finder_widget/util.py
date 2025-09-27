"""
Utility functions for the Eval Finder Widget.
"""

import asyncio
from typing import Any, Dict, List

from softmax.orchestrator.clients.scorecard_client import ScorecardClient


def fetch_eval_data_for_policies(
    training_run_ids: List[str] | None = None,
    run_free_policy_ids: List[str] | None = None,
    category_filter: List[str] | None = None,
    client: ScorecardClient | None = None,
) -> Dict[str, Any]:
    """
    Fetch policy-aware evaluation data for the widget.

    Args:
        training_run_ids: List of training run IDs to get evals for
        run_free_policy_ids: List of standalone policy IDs to get evals for
        category_filter: List of categories to include

    Returns:
        Dictionary with evaluation data ready for widget
    """

    async def _fetch_data():
        # Use provided client or create default one
        scorecard_client = client or ScorecardClient()

        # Use empty lists as defaults
        tr_ids = training_run_ids or []
        rf_ids = run_free_policy_ids or []

        # Get evals that have been run on these policies (only real data from API)
        if tr_ids or rf_ids:
            # print(f"🔍 Querying for training_run_ids: {tr_ids}")
            # print(f"🔍 Querying for run_free_policy_ids: {rf_ids}")
            completed_eval_names = await scorecard_client.get_eval_names(
                training_run_ids=tr_ids, run_free_policy_ids=rf_ids
            )
            # print(
            #     f"🔍 ScorecardClient returned {len(completed_eval_names)} eval names: {completed_eval_names[:5] if completed_eval_names else 'None'}"
            # )

            # Get performance data to understand which evals succeeded/failed
            if completed_eval_names:
                try:
                    scorecard_data = await scorecard_client.generate_scorecard(
                        training_run_ids=tr_ids,
                        run_free_policy_ids=rf_ids,
                        eval_names=completed_eval_names[:10],  # Limit for performance
                        metric="reward",
                        policy_selector="best",
                    )
                    performance_data = scorecard_data.cells
                except Exception:
                    # print(f"Could not fetch performance data: {e}")
                    performance_data = {}
            else:
                performance_data = {}
        else:
            completed_eval_names = []
            performance_data = {}

        # If no policy-specific evals found, fallback to showing all available evaluations
        if not completed_eval_names:
            # print(
            #     "🔍 No policy-specific evaluations found, falling back to all available evaluations"
            # )
            try:
                # Get all available evaluations from the database by querying without policy restrictions
                available_eval_names = await scorecard_client.get_eval_names(
                    training_run_ids=[], run_free_policy_ids=[]
                )
                # print(
                #     f"🔍 Found {len(available_eval_names)} total available evaluations"
                # )
                # Use all available evaluations
                all_evals_to_process = available_eval_names
            except Exception as e:
                print(f"🔍 Could not fetch available evaluations: {e}")
                all_evals_to_process = []
        else:
            # Use policy-specific evaluations
            all_evals_to_process = completed_eval_names

        # Create eval metadata with performance awareness
        evaluations = _create_policy_aware_eval_metadata(
            eval_names=all_evals_to_process,
            completed_evals=completed_eval_names,
            performance_data=performance_data,
            category_filter=category_filter or [],
        )

        # Build category structure
        categories = _build_eval_categories_from_names(
            all_evals_to_process, evaluations
        )

        return {
            "evaluations": evaluations,
            "categories": categories,
            "total_count": len(evaluations),
        }

    # Run the async function - handle both notebook and standalone environments
    try:
        # Check if we're in a notebook environment with an existing event loop
        try:
            asyncio.get_running_loop()
            # We're in a running event loop (Jupyter/Marimo), use nest_asyncio
            import nest_asyncio

            nest_asyncio.apply()
        except RuntimeError:
            # No running loop, which is fine
            pass

        return asyncio.run(_fetch_data())
    except Exception as e:
        print(f"Error fetching eval data: {e}")
        # Return empty data structure on error
        return {
            "evaluations": [],
            "categories": [],
            "total_count": 0,
        }


def _create_policy_aware_eval_metadata(
    eval_names: List[str],
    completed_evals: List[str],
    performance_data: Dict[str, Any],
    category_filter: List[str] | None = None,
) -> List[Dict[str, Any]]:
    """
    Create policy-aware eval metadata from eval names and performance data.
    Uses both heuristics and actual policy performance to enrich metadata.
    """
    evaluations = []

    for eval_name in eval_names:
        if "/" not in eval_name:
            continue

        category, specific_eval = eval_name.split("/", 1)

        # Apply category filter (only real filtering we can do)
        if category_filter and category not in category_filter:
            continue

        # Keep it simple - just use the eval name as description
        # No synthetic tags or status indicators
        base_description = eval_name
        performance_score = None

        # Only track completion status internally, don't show it in UI
        is_completed = eval_name in completed_evals
        if is_completed and performance_data:
            # Get actual performance score if available, but don't add fake tags
            for policy_name, policy_data in performance_data.items():
                # Make sure policy_data is a dict and contains eval_name
                if isinstance(policy_data, dict) and eval_name in policy_data:
                    cell_data = policy_data[eval_name]
                    if hasattr(cell_data, "value"):
                        performance_score = cell_data.value
                    elif isinstance(cell_data, dict) and "value" in cell_data:
                        performance_score = cell_data["value"]
                    break

        evaluations.append(
            {
                "name": eval_name,
                "category": category,
                "description": base_description,
                "difficulty": "unknown",  # API doesn't provide difficulty
                "agent_requirements": [],  # API doesn't provide agent requirements
                "prerequisites": [],  # API doesn't track prerequisites
                "tags": [],  # No synthetic tags
                "is_completed": is_completed,
                "performance_score": performance_score,
            }
        )

    return evaluations


def _build_eval_categories_from_names(
    eval_names: List[str], evaluations: List[Dict[str, Any]]
) -> List[Dict[str, Any]]:
    """Build category structure from eval names and metadata."""
    # print(
    #     f"📂 Building categories from {len(eval_names)} eval names and {len(evaluations)} evaluations"
    # )

    # Group by category
    categories = {}
    eval_lookup = {e["name"]: e for e in evaluations}

    # print(f"📂 Eval names: {eval_names}")
    # print(f"📂 Eval lookup keys: {list(eval_lookup.keys())}")

    for eval_name in eval_names:
        if "/" not in eval_name or eval_name not in eval_lookup:
            continue

        category = eval_name.split("/", 1)[0]
        if category not in categories:
            categories[category] = []
        categories[category].append(eval_name)

    # Build category nodes
    category_nodes = []
    for category, category_evals in categories.items():
        if not category_evals:
            continue

        category_node = {
            "name": category,
            "category": category,
            "children": [],
        }

        for eval_name in category_evals:
            if eval_name in eval_lookup:
                eval_node = {
                    "name": eval_name.split("/")[-1],
                    "category": category,
                    "eval_metadata": eval_lookup[eval_name],
                }
                category_node["children"].append(eval_node)

        if category_node["children"]:  # Only add if has children
            category_nodes.append(category_node)

    # print(f"📂 Built {len(category_nodes)} category nodes")
    # for node in category_nodes:
    #     print(f"📂   - {node['name']}: {len(node['children'])} evaluations")

    return category_nodes


def create_demo_eval_finder_widget():
    """
    Create a demo eval finder widget with sample data.

    Returns:
        EvalFinderWidget instance with demo data loaded
    """
    from .EvalFinderWidget import EvalFinderWidget

    # Create widget
    widget = EvalFinderWidget()

    # Realistic demo data matching what the API actually provides
    demo_evaluations = [
        {
            "name": "navigation/emptyspace_withinsight",
            "category": "navigation",
            "description": "navigation/emptyspace_withinsight",  # API only provides eval name
            "difficulty": "unknown",  # API doesn't classify difficulty
            "agent_requirements": [],  # API doesn't specify requirements
            "prerequisites": [],  # API doesn't track prerequisites
            "tags": ["navigation"],  # Only category from eval name
            "is_completed": True,
        },
        {
            "name": "navigation/labyrinth",
            "category": "navigation",
            "description": "navigation/labyrinth",
            "difficulty": "unknown",
            "agent_requirements": [],
            "prerequisites": [],
            "tags": ["navigation"],
            "is_completed": False,
        },
        {
            "name": "memory/easy",
            "category": "memory",
            "description": "memory/easy",
            "difficulty": "unknown",
            "agent_requirements": [],
            "prerequisites": [],
            "tags": ["memory"],
            "is_completed": True,
        },
        {
            "name": "arena/basic",
            "category": "arena",
            "description": "arena/basic",
            "difficulty": "unknown",
            "agent_requirements": [],
            "prerequisites": [],
            "tags": ["arena"],
            "is_completed": False,
        },
    ]

    # Set demo data - categories will be built automatically from evaluations
    widget.set_eval_data(
        evaluations=demo_evaluations,
    )

    # print("🎯 Demo eval finder widget created with sample data")
    return widget
