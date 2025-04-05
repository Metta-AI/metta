from omegaconf import DictConfig
import dash
from dash import html, dcc
import dash_bootstrap_components as dbc
from dash.dependencies import Input, Output, State

# Import application components
from omegaconf import DictConfig
from components.layout import create_layout
from data.data_loader import load_data
from data.data_processor import process_data
from config import ReportConfig

class App:
    def __init__(self, cfg: ReportConfig):
        self.app = dash.Dash(
            __name__, 
            external_stylesheets=[dbc.themes.BOOTSTRAP],
            suppress_callback_exceptions=True
        )
        self.cfg = cfg
        self.data = load_data(cfg.eval_db_uri, cfg.run_dir)
        self.processed_data = process_data(self.data, cfg)
        self.app.layout = create_layout(self.processed_data)
    def run(self):
        self.app.run_server(debug=self.cfg.debug, port=self.cfg.port)
