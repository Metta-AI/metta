"""
Dashboard application.
"""

import logging
import dash
import dash_bootstrap_components as dbc
from dash import html
from typing import Optional

# Import application components
from .config import DashboardConfig
from .data import load_data, process_data
from sim.eval.viz.layout import create_layout, register_callbacks
from sim.eval.viz.components import register_collapse_callbacks

logger = logging.getLogger(__name__)

class DashboardApp:
    """
    Dashboard application for visualizing RL policy evaluation results.
    """
    
    def __init__(self, cfg: DashboardConfig):
        # Initialize Dash app
        self.app = dash.Dash(
            __name__, 
            external_stylesheets=[dbc.themes.BOOTSTRAP],
            suppress_callback_exceptions=True
        )
        self.app.title = self.cfg.page_title
        
        # Load and process data
        logger.info(f"Loading data from {self.cfg.eval_db_uri}")
        self.data = load_data(self.cfg.eval_db_uri, self.cfg.run_dir)
        logger.info(f"Processing data for visualization")
        self.processed_data = process_data(self.data, self.config_dict)
        
        # Create layout
        self.app.layout = create_layout(self.processed_data, self.config_dict)
        
        # Register callbacks
        self._register_callbacks()
        
        logger.info("Dashboard initialized")
    
    def _register_callbacks(self):
        """Register all dashboard callbacks."""
        # Register collapsible section callbacks
        register_collapse_callbacks(self.app, ["policy-filter", "eval-filter"])
        
        # Register main visualization callbacks
        register_callbacks(self.app, self.processed_data, self.config_dict)
    
    def run(self, debug: Optional[bool] = None, port: Optional[int] = None):
        """
        Run the dashboard server.
        
        Args:
            debug: Enable debug mode (overrides config)
            port: Server port (overrides config)
        """
        debug_mode = debug if debug is not None else self.cfg.debug
        port_number = port if port is not None else self.cfg.port
        
        logger.info(f"Starting dashboard server on port {port_number} (debug={debug_mode})")
        self.app.run_server(debug=debug_mode, port=port_number)
    
    def save_html(self, output_path: Optional[str] = None):
        """
        Save the dashboard as a static HTML file.
        
        Args:
            output_path: Path to save the HTML file (overrides config)
        """
        output_file = output_path if output_path else self.cfg.output_html_path
        
        if not output_file:
            logger.error("No output path specified for HTML export")
            return
        
        try:
            logger.info(f"Saving dashboard to {output_file}")
            
            # Create a copy of the app for HTML generation
            app = self.app
            
            # Get the HTML representation of the app with initial state
            app.layout.children = html.Div([
                html.Link(
                    rel="stylesheet",
                    href="https://stackpath.bootstrapcdn.com/bootstrap/4.5.2/css/bootstrap.min.css"
                ),
                html.Link(
                    rel="stylesheet",
                    href="https://use.fontawesome.com/releases/v5.8.1/css/all.css"
                ),
                app.layout
            ])
            
            # Generate HTML string
            html_string = f'''
            <!DOCTYPE html>
            <html>
                <head>
                    <meta charset="UTF-8">
                    <meta name="viewport" content="width=device-width, initial-scale=1.0">
                    <title>{self.cfg.page_title}</title>
                    <link rel="stylesheet" href="https://stackpath.bootstrapcdn.com/bootstrap/4.5.2/css/bootstrap.min.css">
                    <link rel="stylesheet" href="https://use.fontawesome.com/releases/v5.8.1/css/all.css">
                    <script src="https://cdn.plot.ly/plotly-latest.min.js"></script>
                    <style>
                        .collapsible-section {{
                            margin-bottom: 20px;
                        }}
                        .sidebar {{
                            border-right: 1px solid #eee;
                            padding-right: 20px;
                        }}
                        .dashboard-content {{
                            padding-left: 20px;
                        }}
                    </style>
                </head>
                <body>
                    <div class="container-fluid mt-3">
                        {app._generate_layout_html()}
                    </div>
                    <script src="https://code.jquery.com/jquery-3.5.1.slim.min.js"></script>
                    <script src="https://cdn.jsdelivr.net/npm/bootstrap@4.5.2/dist/js/bootstrap.bundle.min.js"></script>
                    <script>
                        // Simple collapsible section functionality
                        document.addEventListener("DOMContentLoaded", function() {{
                            const collapseBtns = document.querySelectorAll("[id$='-button']");
                            collapseBtns.forEach(btn => {{
                                btn.addEventListener("click", function() {{
                                    const collapseId = this.id.replace("-button", "-collapse");
                                    const collapseElement = document.getElementById(collapseId);
                                    if (collapseElement.style.display === "none") {{
                                        collapseElement.style.display = "block";
                                    }} else {{
                                        collapseElement.style.display = "none";
                                    }}
                                }});
                            }});
                        }});
                    </script>
                </body>
            </html>
            '''
            
            # Write to file
            with open(output_file, 'w', encoding='utf-8') as f:
                f.write(html_string)
            
            logger.info(f"Dashboard saved successfully to {output_file}")
        except Exception as e:
            logger.error(f"Error saving dashboard to HTML: {e}")
            import traceback
            logger.error(traceback.format_exc())