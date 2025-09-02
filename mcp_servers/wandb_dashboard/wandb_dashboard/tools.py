"""
WandB Dashboard Management Tools

Implementation of dashboard creation, modification, and management tools using the WandB Workspace API.
"""

import json
import logging
from typing import Any, Dict, List, Optional

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
        """Create a real WandB dashboard using the WandB API."""
        try:
            logger.info(f"Creating dashboard '{name}' for {entity}/{project}")

            # Use default sections if none provided
            if not sections:
                sections = [
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
                            {"type": "scalar_chart", "config": {"metric": "precision"}},
                            {"type": "scalar_chart", "config": {"metric": "recall"}},
                        ],
                    },
                ]

            # Create the dashboard using the WandB API
            dashboard_result = await self.client.create_dashboard(
                entity=entity, project=project, name=name, description=description, sections=sections
            )

            result = {
                "status": "success",
                "message": f"Dashboard '{name}' created successfully",
                "dashboard": dashboard_result,
            }

            return json.dumps(result, indent=2)

        except Exception as e:
            logger.error(f"Failed to create dashboard: {e}")
            error_result = {"status": "error", "message": f"Failed to create dashboard: {str(e)}"}
            return json.dumps(error_result, indent=2)

    async def update_dashboard(self, dashboard_url: str, modifications: Dict[str, Any]) -> str:
        """Update an existing dashboard using the WandB API."""
        try:
            logger.info(f"Updating dashboard: {dashboard_url}")

            # Update the dashboard using the WandB API
            update_result = await self.client.update_dashboard(dashboard_url, modifications)

            result = {
                "status": "success",
                "message": "Dashboard updated successfully",
                "update": update_result,
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
        """Get the configuration of an existing dashboard using the WandB API."""
        try:
            logger.info(f"Getting config for dashboard: {dashboard_url}")

            # Get the dashboard config using the WandB API
            config = await self.client.get_dashboard_config(dashboard_url)

            result = {"status": "success", "dashboard_config": config}

            return json.dumps(result, indent=2)

        except Exception as e:
            logger.error(f"Failed to get dashboard config: {e}")
            error_result = {"status": "error", "message": f"Failed to get dashboard config: {str(e)}"}
            return json.dumps(error_result, indent=2)

    async def add_panel(
        self, dashboard_url: str, section_name: str, panel_type: str, panel_config: Dict[str, Any]
    ) -> str:
        """Add a panel to an existing dashboard using the WandB API."""
        try:
            logger.info(f"Adding {panel_type} panel to section '{section_name}' in {dashboard_url}")

            # Add the panel using the WandB API
            panel_result = await self.client.add_panel_to_dashboard(
                dashboard_url, section_name, panel_type, panel_config
            )

            result = {
                "status": "success",
                "message": f"Panel added successfully to '{section_name}'",
                "panel": panel_result,
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
        """Clone an existing dashboard using the WandB API."""
        try:
            logger.info(f"Cloning dashboard from {source_url} as '{new_name}'")

            # Clone the dashboard using the WandB API
            clone_result = await self.client.clone_dashboard(source_url, new_name)

            result = {
                "status": "success",
                "message": f"Dashboard cloned successfully as '{new_name}'",
                "clone": clone_result,
            }

            return json.dumps(result, indent=2)

        except Exception as e:
            logger.error(f"Failed to clone dashboard: {e}")
            error_result = {"status": "error", "message": f"Failed to clone dashboard: {str(e)}"}
            return json.dumps(error_result, indent=2)

    async def delete_dashboard(self, dashboard_url: str) -> str:
        """Delete an existing dashboard using the WandB API."""
        try:
            logger.info(f"Deleting dashboard: {dashboard_url}")

            # Delete the dashboard using the WandB API
            delete_result = await self.client.delete_dashboard(dashboard_url)

            result = {
                "status": "success",
                "message": "Dashboard deleted successfully",
                "deleted_dashboard": delete_result,
            }

            return json.dumps(result, indent=2)

        except Exception as e:
            logger.error(f"Failed to delete dashboard: {e}")
            error_result = {"status": "error", "message": f"Failed to delete dashboard: {str(e)}"}
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
                "delete_dashboard",
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
