"""
Configuration for WandB Dashboard MCP Server

Handles server configuration, authentication settings, and default values.
"""

import os
from dataclasses import dataclass, field
from typing import Dict, List, Optional


@dataclass
class WandBMCPConfig:
    """Configuration for the WandB Dashboard MCP Server."""

    # Server configuration
    server_name: str = "wandb-dashboard-mcp"
    version: str = "0.1.0"

    # WandB authentication
    api_key: Optional[str] = field(default_factory=lambda: os.getenv("WANDB_API_KEY"))
    base_url: str = "https://api.wandb.ai"

    # Default entity/project (can be overridden by tools)
    default_entity: Optional[str] = field(default_factory=lambda: os.getenv("WANDB_ENTITY"))
    default_project: Optional[str] = field(default_factory=lambda: os.getenv("WANDB_PROJECT"))

    # Dashboard defaults
    default_x_axis: str = "Step"
    default_smoothing_type: str = "exponential"
    default_smoothing_weight: float = 0.9
    default_max_runs: int = 10

    # Panel type mappings
    supported_panel_types: List[str] = field(
        default_factory=lambda: ["line_plot", "bar_plot", "scalar_chart", "scatter_plot"]
    )

    # Common metrics that are often useful in dashboards
    common_metrics: List[str] = field(
        default_factory=lambda: [
            "loss",
            "accuracy",
            "val_loss",
            "val_accuracy",
            "learning_rate",
            "epoch",
            "step",
            "runtime",
            "f1_score",
            "precision",
            "recall",
        ]
    )

    # Template configurations for common dashboard types
    dashboard_templates: Dict[str, Dict] = field(
        default_factory=lambda: {
            "training_overview": {
                "name": "Training Overview",
                "sections": [
                    {
                        "name": "Loss Metrics",
                        "panels": [
                            {"type": "line_plot", "config": {"x": "Step", "y": ["loss", "val_loss"]}},
                            {"type": "line_plot", "config": {"x": "epoch", "y": ["loss", "val_loss"]}},
                        ],
                    },
                    {
                        "name": "Performance Metrics",
                        "panels": [
                            {"type": "line_plot", "config": {"x": "Step", "y": ["accuracy", "val_accuracy"]}},
                            {"type": "bar_plot", "config": {"metrics": ["f1_score", "precision", "recall"]}},
                        ],
                    },
                ],
            },
            "hyperparameter_analysis": {
                "name": "Hyperparameter Analysis",
                "sections": [
                    {
                        "name": "Learning Rate Analysis",
                        "panels": [
                            {"type": "line_plot", "config": {"x": "Step", "y": ["learning_rate"]}},
                            {"type": "scatter_plot", "config": {"x": "learning_rate", "y": "val_accuracy"}},
                        ],
                    },
                    {
                        "name": "Model Performance",
                        "panels": [
                            {"type": "scalar_chart", "config": {"metric": "val_accuracy", "groupby_aggfunc": "max"}},
                            {"type": "scalar_chart", "config": {"metric": "val_loss", "groupby_aggfunc": "min"}},
                        ],
                    },
                ],
            },
            "model_comparison": {
                "name": "Model Comparison",
                "sections": [
                    {
                        "name": "Validation Metrics",
                        "panels": [
                            {"type": "line_plot", "config": {"x": "epoch", "y": ["val_accuracy"]}},
                            {"type": "line_plot", "config": {"x": "epoch", "y": ["val_loss"]}},
                            {"type": "bar_plot", "config": {"metrics": ["val_accuracy", "val_f1_score"]}},
                        ],
                    }
                ],
            },
        }
    )

    # Logging configuration
    log_level: str = "INFO"
    log_format: str = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"

    @classmethod
    def from_env(cls) -> "WandBMCPConfig":
        """Create configuration from environment variables."""
        return cls(
            api_key=os.getenv("WANDB_API_KEY"),
            default_entity=os.getenv("WANDB_ENTITY"),
            default_project=os.getenv("WANDB_PROJECT"),
            base_url=os.getenv("WANDB_BASE_URL", "https://api.wandb.ai"),
            log_level=os.getenv("LOG_LEVEL", "INFO"),
        )

    def validate(self) -> List[str]:
        """Validate the configuration and return any errors."""
        errors = []

        if not self.api_key:
            errors.append("WANDB_API_KEY not set. Please set the environment variable or provide api_key.")

        if self.default_smoothing_weight < 0 or self.default_smoothing_weight > 1:
            errors.append("default_smoothing_weight must be between 0 and 1")

        if self.default_max_runs <= 0:
            errors.append("default_max_runs must be positive")

        return errors

    def get_template(self, template_name: str) -> Optional[Dict]:
        """Get a dashboard template by name."""
        return self.dashboard_templates.get(template_name)

    def list_templates(self) -> List[str]:
        """List available dashboard template names."""
        return list(self.dashboard_templates.keys())
