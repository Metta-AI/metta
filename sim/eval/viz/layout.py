from dash import html, dcc
import dash_bootstrap_components as dbc
from dash.dependencies import Input, Output, State, ALL
from typing import Dict, Any, List, Tuple, Optional

from .components import (
    create_collapsible_section, 
    create_metric_selector, 
    create_hierarchical_checklist, 
    register_search_callbacks
)

from .dashboards import (
    create_matrix_visualization, 
)

def create_filters_section(processed_data: Dict[str, Any]) -> html.Div:
    """
    Create the filters section of the dashboard.
    
    Args:
        processed_data: Processed data dictionary containing hierarchies and display maps
        
    Returns:
        Dash layout component for filters
    """
    # Create metric selector
    metric_selector = create_metric_selector(
        id_prefix="dashboard",
        metrics=processed_data["metrics"],
        default_metric=next((m for m in processed_data["metrics"] if "episode_reward" in m), None)
    )
    
    # Create policy filter
    policy_filter = create_collapsible_section(
        title="Filter Policies",
        content=create_hierarchical_checklist(
            id_prefix="policy",
            hierarchy=processed_data["policy_hierarchy"],
            display_names=processed_data["policy_display_names"]
        ),
        id_prefix="policy-filter",
        is_open=True
    )
    
    # Create eval filter
    eval_filter = create_collapsible_section(
        title="Filter Evaluations",
        content=create_hierarchical_checklist(
            id_prefix="eval",
            hierarchy=processed_data["eval_hierarchy"],
            display_names=processed_data["eval_display_names"]
        ),
        id_prefix="eval-filter",
        is_open=True
    )
    
    # Return complete filters section
    return html.Div([
        html.H3("Filters", className="mb-3"),
        metric_selector,
        policy_filter,
        eval_filter
    ], className="filters-section mb-4")

def create_dashboard_content(
    processed_data: Dict[str, Any], 
    config: Dict[str, Any],
    filtered_policies: Optional[List[str]] = None,
    filtered_evals: Optional[List[str]] = None,
    selected_metric: Optional[str] = None
) -> html.Div:  
    """
    Create the main dashboard content with visualizations.
    
    Args:
        processed_data: Processed data dictionary
        config: Application configuration
        filtered_policies: Optional list of policy IDs to include
        filtered_evals: Optional list of evaluation IDs to include
        selected_metric: Optional selected metric, defaults to episode_reward if available
        
    Returns:
        Dash layout component for dashboard content
    """
    # Default metric selection if not provided
    if not selected_metric:
        selected_metric = next(
            (m for m in processed_data["metrics"] if "episode_reward" in m),
            processed_data["metrics"][0] if processed_data["metrics"] else None
        )
    
    # Create visualizations
    visualizations = []
    
    # Matrix visualization for the selected metric
    if selected_metric and f'{selected_metric}_matrix' in processed_data:
        matrix_viz = create_matrix_visualization(
            processed_data[f'{selected_metric}_matrix'],
            config,
            selected_metric,
            filtered_policies=filtered_policies,
            filtered_evals=filtered_evals
        )
        visualizations.append(matrix_viz)
        
    # Return content container
    return html.Div(visualizations, className="dashboard-content")

def create_layout(processed_data: Dict[str, Any], config: Dict[str, Any] = None) -> html.Div:
    """
    Create the main dashboard layout.
    
    Args:
        processed_data: Processed data dictionary
        config: Application configuration
        
    Returns:
        Dash layout component for the entire dashboard
    """
    if config is None:
        config = {
            'default_graph_height': 500,
            'default_graph_width': 800,
            'pass_threshold': 2.95,
            'matrix_colorscale': 'RdYlGn',
            'matrix_score_range': (0, 3),
            'page_title': "Metta Policy Evaluation Dashboard"
        }
    
    return html.Div([
        # External stylesheets
        html.Link(
            rel="stylesheet",
            href="https://use.fontawesome.com/releases/v5.8.1/css/all.css"
        ),
                
        # Main content container
        dbc.Container([
            dbc.Row([
                # Sidebar with filters
                dbc.Col(
                    create_filters_section(processed_data),
                    width=3,
                    className="sidebar"
                ),
                
                # Main content area
                dbc.Col(
                    create_dashboard_content(processed_data, config),
                    width=9,
                    className="main-content"
                )
            ], className="dashboard-row")
        ], fluid=True, className="dashboard-container")
    ], className="dashboard")

def get_category_ids(processed_data: Dict[str, Any], prefix: str) -> List[str]:
    """
    Get all category checklist IDs for a specific hierarchy.
    
    Args:
        processed_data: Processed data dictionary
        prefix: Prefix for the hierarchy (e.g., 'policy' or 'eval')
        
    Returns:
        List of category checklist IDs
    """
    hierarchy_key = f"{prefix}_hierarchy"
    if hierarchy_key not in processed_data:
        return []
    
    return [
        f"{prefix}-{category.lower().replace(' ', '-')}-checklist"
        for category in processed_data[hierarchy_key].keys()
    ]

def register_collapse_and_search_callbacks(app, processed_data: Dict[str, Any]) -> Tuple[List[str], List[str]]:
    """
    Register all basic UI callbacks (collapse and search).
    
    Args:
        app: Dash application instance
        processed_data: Processed data dictionary
        
    Returns:
        Tuple of (policy_category_ids, eval_category_ids)
    """
    # Get all category IDs (skip registering collapsible section callbacks here)
    policy_category_ids = get_category_ids(processed_data, "policy")
    eval_category_ids = get_category_ids(processed_data, "eval")
    
    # Register search callbacks for policies
    if "policy_hierarchy" in processed_data:
        register_search_callbacks(
            app, 
            "policy", 
            processed_data["policy_hierarchy"], 
            processed_data["policy_display_names"]
        )
    
    # Register search callbacks for evaluations
    if "eval_hierarchy" in processed_data:
        register_search_callbacks(
            app, 
            "eval", 
            processed_data["eval_hierarchy"], 
            processed_data["eval_display_names"]
        )
    
    return policy_category_ids, eval_category_ids

def register_visualization_callbacks(
    app, 
    processed_data: Dict[str, Any], 
    config: Dict[str, Any],
    policy_category_ids: List[str],
    eval_category_ids: List[str]
):
    """
    Register visualization update callbacks.
    
    Args:
        app: Dash application instance
        processed_data: Processed data dictionary
        config: Application configuration
        policy_category_ids: List of policy category checklist IDs
        eval_category_ids: List of evaluation category checklist IDs
    """
    # Callback to update visualizations based on filters and metric selection
    @app.callback(
        Output("dashboard-content", "children"),
        [
            Input("dashboard-metric-selector", "value"),
            *[Input(id, "value") for id in policy_category_ids],
            *[Input(id, "value") for id in eval_category_ids]
        ]
    )
    def update_visualizations(metric, *selected_items):
        """
        Update visualizations based on selected metric and filter values.
        
        Args:
            metric: Selected metric
            *selected_items: Lists of selected items from each category
            
        Returns:
            Updated dashboard content with filtered visualizations
        """
        # Extract policy selections
        policy_categories = list(processed_data["policy_hierarchy"].keys())
        num_policy_categories = len(policy_categories)
        
        # Combine all selected policies across categories
        selected_policies = []
        for i, category in enumerate(policy_categories):
            if i < len(selected_items):
                category_selections = selected_items[i]
                if category_selections:
                    selected_policies.extend(category_selections)
        
        # Extract evaluation selections
        eval_categories = list(processed_data["eval_hierarchy"].keys())
        
        # Combine all selected evaluations across categories
        selected_evals = []
        for i, category in enumerate(eval_categories):
            idx = i + num_policy_categories
            if idx < len(selected_items):
                category_selections = selected_items[idx]
                if category_selections:
                    selected_evals.extend(category_selections)
        
        # Create visualizations with filters applied
        visualizations = []
        
        # Matrix visualization for the selected metric
        if metric and f'{metric}_matrix' in processed_data:
            matrix_viz = create_matrix_visualization(
                processed_data[f'{metric}_matrix'],
                config,
                metric,
                filtered_policies=selected_policies if selected_policies else None,
                filtered_evals=selected_evals if selected_evals else None
            )
            visualizations.append(matrix_viz)
        
        return visualizations

def register_callbacks(app, processed_data: Dict[str, Any], config: Dict[str, Any]):
    """
    Register all callbacks for the dashboard.
    
    Args:
        app: Dash application instance
        processed_data: Processed data dictionary
        config: Application configuration
    """
    # Register basic UI callbacks and get category IDs
    policy_category_ids, eval_category_ids = register_collapse_and_search_callbacks(app, processed_data)
    
    # Register visualization callbacks
    register_visualization_callbacks(app, processed_data, config, policy_category_ids, eval_category_ids)