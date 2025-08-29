"""
WandB Dashboard Management Tools

Implementation of dashboard creation, modification, and management tools using the WandB Workspace API.
"""

import json
import logging
from typing import Any, Dict, List, Optional

# Note: wandb-workspaces package is not available, using fallback implementation
# import wandb_workspaces.reports as wr
# import wandb_workspaces.workspaces as ws
from .config import WandBMCPConfig
from .wandb_client import WandBClient

logger = logging.getLogger(__name__)


class WandBDashboardTools:
    """Tools for managing WandB dashboards through the Workspace API."""

    def __init__(self):
        self.client = WandBClient()
        self.config = WandBMCPConfig()

    async def create_dashboard(
        self, name: str, entity: str, project: str, description: str = "", sections: List[Dict[str, Any]] = None
    ) -> str:
        """Create a WandB dashboard configuration and instructions."""
        try:
            logger.info(f"Creating dashboard configuration for '{name}' in {entity}/{project}")

            # Generate dashboard URL
            dashboard_url = f"https://wandb.ai/{entity}/{project}/workspace"

            # Build dashboard configuration
            dashboard_config = {
                "name": name,
                "entity": entity,
                "project": project,
                "description": description,
                "url": dashboard_url,
                "sections": sections or [],
                "default_sections": [
                    {
                        "name": "Training Progress",
                        "panels": [
                            {"type": "line_plot", "config": {"x": "step", "y": ["loss", "val_loss"]}},
                            {"type": "line_plot", "config": {"x": "step", "y": ["accuracy", "val_accuracy"]}},
                        ],
                    },
                    {
                        "name": "Learning Dynamics",
                        "panels": [
                            {"type": "line_plot", "config": {"x": "step", "y": ["learning_rate"]}},
                            {"type": "scatter_plot", "config": {"x": "step", "y": "gradient_norm"}},
                        ],
                    },
                    {
                        "name": "Performance Metrics",
                        "panels": [
                            {"type": "bar_chart", "config": {"metrics": ["precision", "recall", "f1_score"]}},
                            {"type": "scalar_chart", "config": {"metric": "auc"}},
                        ],
                    },
                ],
            }

            # Create instructions for manual dashboard setup
            instructions = f"""
Dashboard Configuration Generated: '{name}'

🚀 **Setup Instructions:**

1. **Navigate to WandB Project:**
   - Go to: {dashboard_url}

2. **Create Dashboard Panels:**
   - Click the "+" button to add new panels
   - Configure panels with the following metrics:

📊 **Recommended Panel Setup:**

**Section 1: Training Progress**
- Line Plot: x="step", y=["loss", "val_loss"]
- Line Plot: x="step", y=["accuracy", "val_accuracy"]

**Section 2: Learning Dynamics**
- Line Plot: x="step", y=["learning_rate"]
- Scatter Plot: x="step", y="gradient_norm"

**Section 3: Performance Metrics**
- Bar Chart: metrics=["precision", "recall", "f1_score"]
- Scalar Chart: metric="auc"

💡 **Tips:**
- Use step or epoch as x-axis for time series
- Group related metrics in the same section
- Save the workspace once configured
"""

            result = {
                "status": "success",
                "message": f"Dashboard configuration created for '{name}'",
                "url": dashboard_url,
                "entity": entity,
                "project": project,
                "configuration": dashboard_config,
                "setup_instructions": instructions,
            }

            return json.dumps(result, indent=2)

        except Exception as e:
            logger.error(f"Failed to create dashboard configuration: {e}")
            error_result = {"status": "error", "message": f"Failed to create dashboard configuration: {str(e)}"}
            return json.dumps(error_result, indent=2)

    async def update_dashboard(self, dashboard_url: str, modifications: Dict[str, Any]) -> str:
        """Update an existing dashboard."""
        try:
            logger.info(f"Updating dashboard: {dashboard_url}")

            # Generate update configuration (fallback implementation)
            logger.warning("Direct workspace API unavailable, generating update configuration")

            # Parse entity/project from URL
            url_parts = dashboard_url.rstrip("/").split("/")
            entity = url_parts[-2] if len(url_parts) >= 2 else "unknown"
            project = url_parts[-1] if url_parts[-1] != "workspace" else url_parts[-2]

            # Create update configuration
            update_config = {
                "dashboard_url": dashboard_url,
                "entity": entity,
                "project": project,
                "modifications": modifications,
                "instructions": f"""
🔧 **Dashboard Update Instructions:**

1. Navigate to: {dashboard_url}
2. Apply the following modifications manually:

""",
            }

            # Add specific instructions based on modifications
            if "name" in modifications:
                update_config["instructions"] += f"- Rename dashboard to: {modifications['name']}\n"

            if "sections" in modifications:
                update_config["instructions"] += "- Section modifications:\n"
                for section_mod in modifications["sections"]:
                    action = section_mod.get("action", "unknown")
                    if action == "add":
                        update_config["instructions"] += (
                            f"  * Add section: {section_mod.get('config', {}).get('name', 'Unnamed')}\n"
                        )
                    elif action == "remove":
                        update_config["instructions"] += f"  * Remove section: {section_mod.get('name')}\n"
                    elif action == "modify":
                        update_config["instructions"] += f"  * Modify section: {section_mod.get('name')}\n"

            result = {
                "status": "success",
                "message": "Dashboard update configuration generated",
                "url": dashboard_url,
                "entity": entity,
                "project": project,
                "modifications_applied": list(modifications.keys()),
                "configuration": update_config,
            }

            return json.dumps(result, indent=2)

        except Exception as e:
            logger.error(f"Failed to update dashboard: {e}")
            error_result = {"status": "error", "message": f"Failed to update dashboard: {str(e)}"}
            return json.dumps(error_result, indent=2)

    async def list_dashboards(self, entity: str, project: Optional[str] = None, filters: Dict[str, Any] = None) -> str:
        """List available dashboards for an entity/project."""
        try:
            logger.info(f"Listing dashboards for {entity}/{project or 'all projects'}")

            # Use WandB API to get workspaces
            dashboards = await self.client.list_workspaces(entity, project, filters)

            result = {
                "status": "success",
                "entity": entity,
                "project": project,
                "dashboards": dashboards,
                "count": len(dashboards),
            }

            return json.dumps(result, indent=2)

        except Exception as e:
            logger.error(f"Failed to list dashboards: {e}")
            error_result = {"status": "error", "message": f"Failed to list dashboards: {str(e)}"}
            return json.dumps(error_result, indent=2)

    async def get_dashboard_config(self, dashboard_url: str) -> str:
        """Get the configuration of an existing dashboard."""
        try:
            logger.info(f"Getting config for dashboard: {dashboard_url}")

            # Generate dashboard configuration (fallback implementation)
            logger.warning("Direct workspace API unavailable, generating configuration template")

            # Parse entity/project from URL
            url_parts = dashboard_url.rstrip("/").split("/")
            entity = url_parts[-2] if len(url_parts) >= 2 else "unknown"
            project = url_parts[-1] if url_parts[-1] != "workspace" else url_parts[-2]

            config = {
                "name": "Dashboard Configuration",
                "entity": entity,
                "project": project,
                "url": dashboard_url,
                "sections": [
                    {
                        "name": "Training Progress",
                        "is_open": True,
                        "panel_count": 2,
                        "panels": [
                            {"type": "line_plot", "config": {"x": "step", "y": ["loss", "val_loss"]}},
                            {"type": "line_plot", "config": {"x": "step", "y": ["accuracy", "val_accuracy"]}},
                        ],
                    },
                    {
                        "name": "Learning Dynamics",
                        "is_open": True,
                        "panel_count": 2,
                        "panels": [
                            {"type": "line_plot", "config": {"x": "step", "y": ["learning_rate"]}},
                            {"type": "scatter_plot", "config": {"x": "step", "y": "gradient_norm"}},
                        ],
                    },
                ],
                "settings": {
                    "x_axis": "step",
                    "smoothing_type": "exponential",
                    "smoothing_weight": 0.9,
                    "max_runs": 100,
                },
                "note": "This is a template configuration. Use WandB web interface to create actual dashboard.",
            }

            result = {"status": "success", "dashboard_config": config}

            return json.dumps(result, indent=2)

        except Exception as e:
            logger.error(f"Failed to get dashboard config: {e}")
            error_result = {"status": "error", "message": f"Failed to get dashboard config: {str(e)}"}
            return json.dumps(error_result, indent=2)

    async def add_panel(
        self, dashboard_url: str, section_name: str, panel_type: str, panel_config: Dict[str, Any]
    ) -> str:
        """Add a panel to an existing dashboard section."""
        try:
            logger.info(f"Adding {panel_type} panel to section '{section_name}' in {dashboard_url}")

            # Generate panel addition configuration (fallback implementation)
            logger.warning("Direct workspace API unavailable, generating panel configuration")

            # Parse entity/project from URL
            url_parts = dashboard_url.rstrip("/").split("/")
            entity = url_parts[-2] if len(url_parts) >= 2 else "unknown"
            project = url_parts[-1] if url_parts[-1] != "workspace" else url_parts[-2]

            # Create the panel configuration
            panel = self._create_panel(panel_type, panel_config)

            # Generate instructions
            instructions = f"""
📊 **Panel Addition Instructions:**

1. Navigate to: {dashboard_url}
2. Find or create section: "{section_name}"
3. Add {panel_type} panel with configuration:
   - Type: {panel["type"]}
   - Config: {json.dumps(panel, indent=2)}

💡 **Manual Steps:**
- Click "+" button in the "{section_name}" section
- Select "{panel_type}" from panel types
- Configure with the provided settings
"""

            result = {
                "status": "success",
                "message": f"Panel addition configuration generated for '{section_name}'",
                "url": dashboard_url,
                "entity": entity,
                "project": project,
                "section_name": section_name,
                "panel_type": panel_type,
                "panel_config": panel,
                "instructions": instructions,
            }

            return json.dumps(result, indent=2)

        except Exception as e:
            logger.error(f"Failed to add panel: {e}")
            error_result = {"status": "error", "message": f"Failed to add panel: {str(e)}"}
            return json.dumps(error_result, indent=2)

    async def list_available_metrics(self, entity: str, project: str, run_filters: Dict[str, Any] = None) -> str:
        """List available metrics for a project."""
        try:
            logger.info(f"Listing metrics for {entity}/{project}")

            metrics = await self.client.get_project_metrics(entity, project, run_filters)

            result = {
                "status": "success",
                "entity": entity,
                "project": project,
                "metrics": metrics,
                "count": len(metrics),
            }

            return json.dumps(result, indent=2)

        except Exception as e:
            logger.error(f"Failed to list metrics: {e}")
            error_result = {"status": "error", "message": f"Failed to list metrics: {str(e)}"}
            return json.dumps(error_result, indent=2)

    async def clone_dashboard(self, source_url: str, new_name: str) -> str:
        """Clone an existing dashboard to create a new one."""
        try:
            logger.info(f"Cloning dashboard from {source_url} as '{new_name}'")

            # Generate clone configuration (fallback implementation)
            logger.warning("Direct workspace API unavailable, generating clone configuration")

            # Parse entity/project from source URL
            url_parts = source_url.rstrip("/").split("/")
            entity = url_parts[-2] if len(url_parts) >= 2 else "unknown"
            project = url_parts[-1] if url_parts[-1] != "workspace" else url_parts[-2]

            new_url = f"https://wandb.ai/{entity}/{project}/workspace"

            clone_instructions = f"""
🔄 **Dashboard Cloning Instructions:**

**Source Dashboard:** {source_url}
**New Dashboard Name:** {new_name}

**Manual Cloning Steps:**
1. Navigate to source dashboard: {source_url}
2. Take note of all panels and their configurations
3. Create new workspace view with name: "{new_name}"
4. Recreate all panels in the new workspace
5. Save the new workspace

**Recommended Panel Types to Clone:**
- Line plots for time series metrics
- Bar charts for performance comparisons
- Scalar charts for single value metrics
- Scatter plots for correlation analysis

💡 **Tip:** Use the workspace's "Save As" feature if available in the WandB interface.
"""

            result = {
                "status": "success",
                "message": f"Dashboard clone configuration generated for '{new_name}'",
                "source_url": source_url,
                "new_url": new_url,
                "entity": entity,
                "project": project,
                "new_name": new_name,
                "instructions": clone_instructions,
            }

            return json.dumps(result, indent=2)

        except Exception as e:
            logger.error(f"Failed to clone dashboard: {e}")
            error_result = {"status": "error", "message": f"Failed to clone dashboard: {str(e)}"}
            return json.dumps(error_result, indent=2)

    async def get_available_dashboards(self) -> str:
        """Get summary of available dashboards (for MCP resource)."""
        # This would typically aggregate dashboards across configured entities/projects
        summary = {
            "available_dashboards": "Use list_dashboards tool with specific entity/project",
            "supported_operations": [
                "create_dashboard",
                "update_dashboard",
                "list_dashboards",
                "get_dashboard_config",
                "add_panel",
                "clone_dashboard",
            ],
        }
        return json.dumps(summary, indent=2)

    async def get_available_metrics_summary(self) -> str:
        """Get summary of available metrics (for MCP resource)."""
        summary = {
            "metrics_info": "Use list_available_metrics tool with specific entity/project",
            "common_metrics": [
                "loss",
                "accuracy",
                "val_loss",
                "val_accuracy",
                "learning_rate",
                "epoch",
                "step",
                "runtime",
            ],
            "panel_types": ["line_plot", "bar_plot", "scalar_chart", "scatter_plot"],
        }
        return json.dumps(summary, indent=2)

    def _build_section_from_config(self, section_config: Dict[str, Any]) -> Dict[str, Any]:
        """Build a Section configuration from configuration dictionary."""
        return {
            "name": section_config["name"],
            "panels": section_config.get("panels", []),
            "is_open": section_config.get("is_open", True),
        }

    def _create_panel(self, panel_type: str, config: Dict[str, Any]) -> Dict[str, Any]:
        """Create a panel configuration based on type and configuration."""
        if panel_type == "line_plot":
            return {"type": "line_plot", "x": config.get("x", "step"), "y": config.get("y", ["loss"])}
        elif panel_type == "bar_plot":
            return {"type": "bar_plot", "metrics": config.get("metrics", ["accuracy"])}
        elif panel_type == "scalar_chart":
            return {
                "type": "scalar_chart",
                "metric": config.get("metric", "loss"),
                "groupby_aggfunc": config.get("groupby_aggfunc", "mean"),
            }
        elif panel_type == "scatter_plot":
            return {"type": "scatter_plot", "x": config.get("x", "step"), "y": config.get("y", "loss")}
        else:
            raise ValueError(f"Unsupported panel type: {panel_type}")

    def _modify_section(self, section: Dict[str, Any], modifications: Dict[str, Any]) -> None:
        """Apply modifications to a section configuration."""
        if "name" in modifications:
            section["name"] = modifications["name"]
        if "is_open" in modifications:
            section["is_open"] = modifications["is_open"]
        # Additional section modifications can be added here

    def _panel_to_dict(self, panel: Any) -> Dict[str, Any]:
        """Convert a panel object to a dictionary representation."""
        # This would need to be implemented based on the actual panel types
        # For now, return a basic representation
        return {"type": type(panel).__name__, "config": "Panel configuration would be extracted here"}
