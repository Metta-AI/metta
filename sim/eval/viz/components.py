
import json
import dash
from dash import html, dcc
import dash_bootstrap_components as dbc
from dash.dependencies import Input, Output, State, ALL
from typing import Any, List, Dict, Optional, Callable, Set, Tuple

def create_collapsible_section(
    title: str,
    content: Any,
    id_prefix: str,
    is_open: bool = True
) -> html.Div:
    collapse_id = f"{id_prefix}-collapse"
    button_id = f"{id_prefix}-button"
    
    return html.Div([
        html.Div([
            html.H5(
                dbc.Button(
                    [
                        html.Span(title),
                        html.I(className="fas fa-chevron-down ms-2")
                    ],
                    id=button_id,
                    color="link",
                    className="text-decoration-none p-0 text-left d-flex align-items-center"
                ),
                className="mb-0"
            ),
            dbc.Collapse(
                content,
                id=collapse_id,
                is_open=is_open
            )
        ], className="mb-3")
    ])

def toggle_collapse(n_clicks: int, is_open: bool) -> bool:
    if n_clicks:
        return not is_open
    return is_open

def register_collapse_callbacks(app, id_prefixes: List[str]):
    for prefix in id_prefixes:
        collapse_id = f"{prefix}-collapse"
        button_id = f"{prefix}-button"
        
        app.callback(
            Output(collapse_id, "is_open"),
            [Input(button_id, "n_clicks")],
            [State(collapse_id, "is_open")],
        )(toggle_collapse)

def create_filter_controls(
    id_prefix: str,
    items: List[str],
    placeholder: str = "Filter...",
    multi: bool = False
) -> html.Div:
    dropdown_id = f"{id_prefix}-filter"
    
    return html.Div([
        dcc.Dropdown(
            id=dropdown_id,
            options=[{"label": item, "value": item} for item in items],
            placeholder=placeholder,
            multi=multi,
            className="mb-3"
        )
    ])

def create_search_box(
    id_prefix: str,
    placeholder: str = "Search...",
    className: str = "mb-2"
) -> html.Div:
    search_id = f"{id_prefix}-search"
    results_id = f"{id_prefix}-search-results"
    
    return html.Div([
        dcc.Input(
            id=search_id,
            type="text",
            placeholder=placeholder,
            className=f"form-control {className}",
            style={"width": "100%"}
        ),
        html.Div(id=results_id, className="mb-2")
    ])

def create_category_section(
    category: str,
    items: List[str],
    id_prefix: str,
    display_names: Optional[Dict[str, str]] = None,
    is_open: bool = False
) -> html.Div:
    category_id = f"{id_prefix}-{category.lower().replace(' ', '-')}"
    
    # Create checklist options
    options = []
    for item in items:
        display_name = display_names.get(item, item) if display_names else item
        options.append({"label": display_name, "value": item})
    
    # Create checklist for this category
    checklist = dbc.Checklist(
        id=f"{category_id}-checklist",
        options=options,
        value=[],  # Initially none selected
        inline=False
    )
    
    # Create collapsible section
    return create_collapsible_section(
        title=category.title(),
        content=checklist,
        id_prefix=category_id,
        is_open=is_open
    )

def create_hierarchical_checklist(
    id_prefix: str,
    hierarchy: Dict[str, List[str]],
    display_names: Optional[Dict[str, str]] = None
) -> html.Div:
    # Create search box
    search_box = create_search_box(
        id_prefix=id_prefix,
        placeholder="Search items...",
        className="mb-3"
    )
    
    # Create sections for each category
    sections = []
    for category, items in hierarchy.items():
        section = create_category_section(
            category=category,
            items=items,
            id_prefix=id_prefix,
            display_names=display_names,
            is_open=False
        )
        sections.append(section)
    
    # Return complete component
    return html.Div([
        search_box,
        html.Div(sections, id=f"{id_prefix}-categories")
    ])

def create_metric_selector(
    id_prefix: str,
    metrics: List[str],
    default_metric: Optional[str] = None
) -> html.Div:
    dropdown_id = f"{id_prefix}-metric-selector"
    
    # Format metric names for display
    options = [
        {"label": metric.replace("_", " ").title(), "value": metric}
        for metric in metrics
    ]
    
    # Set default value
    value = default_metric if default_metric in metrics else metrics[0] if metrics else None
    
    return html.Div([
        html.Label("Select Metric:"),
        dcc.Dropdown(
            id=dropdown_id,
            options=options,
            value=value,
            clearable=False,
            className="mb-3"
        )
    ])

def flat_item_list(
    hierarchy: Dict[str, List[str]],
    display_names: Optional[Dict[str, str]] = None
) -> List[Dict[str, Any]]:
    all_items = []
    for category, items in hierarchy.items():
        category_id = category.lower().replace(' ', '-')
        for item in items:
            display_name = display_names.get(item, item) if display_names else item
            all_items.append({
                "id": item, 
                "display": display_name, 
                "category": category,
                "category_id": category_id
            })
    
    # Sort by display name for predictable ordering
    all_items.sort(key=lambda x: x["display"])
    return all_items

def register_search_callbacks(
    app,
    id_prefix: str, 
    hierarchy: Dict[str, List[str]],
    display_names: Optional[Dict[str, str]] = None
):
    search_id = f"{id_prefix}-search"
    results_id = f"{id_prefix}-search-results"
    
    # Create flat list of all items for efficient search
    all_items = flat_item_list(hierarchy, display_names)
    
    # Callback to update search results
    @app.callback(
        Output(results_id, "children"),
        [Input(search_id, "value")]
    )
    def update_search_results(search_term):
        if not search_term:
            return []
        
        # Filter items based on search term (case-insensitive)
        search_term = search_term.lower()
        filtered_items = [
            item for item in all_items 
            if search_term in item["display"].lower() or search_term in item["id"].lower()
        ]
        
        if not filtered_items:
            return html.Div("No matches found", className="text-muted small")
        
        # Create quick select buttons for matching items (limit to 10 for performance)
        buttons = []
        for item in filtered_items[:10]:
            buttons.append(
                dbc.Button(
                    item["display"],
                    id={"type": f"{id_prefix}-search-select", "index": item["id"]},
                    color="primary",
                    size="sm",
                    className="me-1 mb-1"
                )
            )
        
        return html.Div([
            html.Div("Matching items:", className="text-muted small mb-1"),
            html.Div(buttons)
        ])
    
    # Create output list for category checklists
    outputs = [
        Output(f"{id_prefix}-{category.lower().replace(' ', '-')}-checklist", "value")
        for category in hierarchy.keys()
    ]
    
    # Create state list for current checklist values
    states = [
        State(f"{id_prefix}-{category.lower().replace(' ', '-')}-checklist", "value")
        for category in hierarchy.keys()
    ]
    
    # Callback to handle selecting an item from search results
    @app.callback(
        outputs,
        [Input({"type": f"{id_prefix}-search-select", "index": ALL}, "n_clicks")],
        states
    )
    def select_search_item(clicks, *current_values):
        ctx = dash.callback_context
        if not ctx.triggered or not any(c for c in clicks if c):
            return current_values
        
        # Find which button was clicked
        button_id = ctx.triggered[0]['prop_id'].split('.')[0]
        item_id = json.loads(button_id)["index"]
        
        # Find which category this item belongs to
        item_category = None
        for category, items in hierarchy.items():
            if item_id in items:
                item_category = category
                break
        
        if item_category is None:
            return current_values
        
        # Update the values for the appropriate category
        category_idx = list(hierarchy.keys()).index(item_category)
        new_values = list(current_values)
        
        # Add item to selection if not already there
        if item_id not in new_values[category_idx]:
            new_values[category_idx] = new_values[category_idx] + [item_id]
        
        return new_values