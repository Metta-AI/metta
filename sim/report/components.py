def create_collapsible_section(
    title: str,
    content: Any,
    id_prefix: str,
    is_open: bool = True
) -> html.Div:
    """
    Create a collapsible section component.
    
    Args:
        title: Section title
        content: Content to display in the section
        id_prefix: Prefix for component IDs
        is_open: Whether the section is initially open
        
    Returns:
        Dash layout component for a collapsible section
    """
    collapse_id = f"{id_prefix}-collapse"
    button_id = f"{id_prefix}-button"
    
    return html.Div([
        html.Div([
            html.H5(
                dbc.Button(
                    [
                        html.Span(title),
                        html.I(className="fa fa-chevron-down ms-2")
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

def register_collapse_callbacks(app, id_prefixes: List[str]):
    """
    Register callbacks for collapsible sections.
    
    Args:
        app: Dash application instance
        id_prefixes: List of ID prefixes for collapsible sections
    """
    for prefix in id_prefixes:
        collapse_id = f"{prefix}-collapse"
        button_id = f"{prefix}-button"
        
        app.callback(
            Output(collapse_id, "is_open"),
            [Input(button_id, "n_clicks")],
            [State(collapse_id, "is_open")],
        )(toggle_collapse)

def toggle_collapse(n_clicks, is_open):
    """
    Toggle collapse state for a collapsible section.
    
    Args:
        n_clicks: Number of clicks on the toggle button
        is_open: Current open state
        
    Returns:
        New open state
    """
    if n_clicks:
        return not is_open
    return is_open