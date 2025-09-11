"""
WandB API Client Wrapper

Wrapper around the WandB API for dashboard and metrics management.
Handles authentication, error handling, and provides convenient methods for MCP tools.
"""

import json
import logging
from typing import Any, Dict, List, Optional

import wandb
import wandb_workspaces.reports.v2 as wr
from wandb import Api

logger = logging.getLogger(__name__)


class WandBClient:
    """Client wrapper for WandB API operations."""

    def __init__(self, api_key: Optional[str] = None):
        """Initialize the WandB client.

        Args:
            api_key: Optional API key. If not provided, will use environment variable or login.
        """
        self.api = None
        self.api_key = api_key
        self._initialize_api()

    def _initialize_api(self) -> None:
        """Initialize the WandB API client."""
        try:
            # Try to use existing wandb authentication first
            try:
                self.api = Api()
                # Test the API by getting viewer info
                _ = self.api.viewer
                logger.info("WandB API client initialized successfully using cached credentials")
                return
            except Exception:
                logger.info("Cached credentials not working, trying explicit login...")

            # If cached credentials don't work, try explicit login
            if self.api_key:
                wandb.login(key=self.api_key)
                self.api = Api()
                logger.info("WandB API client initialized successfully with provided API key")
            else:
                logger.warning("No API key provided and cached credentials failed")
                # Still try to initialize API in case wandb is configured
                self.api = Api()
                logger.info("WandB API client initialized (authentication may be limited)")

        except Exception as e:
            logger.error(f"Failed to initialize WandB API: {e}")
            raise

    def _create_fresh_api_client(self) -> Api:
        """Create a fresh API client to bypass caching issues."""
        try:
            fresh_api = Api()
            # Test the API
            _ = fresh_api.viewer
            logger.info("Created fresh WandB API client")
            return fresh_api
        except Exception as e:
            logger.error(f"Failed to create fresh API client: {e}")
            # Fall back to existing client
            return self.api

    def _parse_dashboard_url(self, dashboard_url: str) -> tuple:
        """Parse WandB dashboard URL to extract entity, project, and report_id.

        Args:
            dashboard_url: WandB dashboard URL

        Returns:
            tuple: (entity, project, report_id)
        """
        # URL format: https://wandb.ai/{entity}/{project}/reports/{report_name}--{report_id}
        if "wandb.ai" not in dashboard_url:
            raise ValueError(f"Invalid WandB URL format: {dashboard_url}")

        # Extract components from URL
        url_parts = dashboard_url.replace("https://wandb.ai/", "").split("/")
        if len(url_parts) < 3 or "reports" not in url_parts:
            raise ValueError(f"Cannot parse dashboard URL: {dashboard_url}")

        entity = url_parts[0]
        project = url_parts[1]

        # Extract report ID from the last part (after the double dash)
        report_part = url_parts[-1]  # e.g., "Report-Name--VmlldzoxNDE4ODY1Ng"
        if "--" not in report_part:
            raise ValueError(f"Cannot extract report ID from URL: {dashboard_url}")

        report_id = report_part.split("--")[-1]

        return entity, project, report_id

    async def list_workspaces(
        self, entity: str, project: Optional[str] = None, filters: Optional[Dict[str, Any]] = None
    ) -> List[Dict[str, Any]]:
        """List available workspaces/dashboards for an entity/project.

        Args:
            entity: WandB entity (user or team)
            project: Optional project name. If None, lists across all projects.
            filters: Optional filters to apply

        Returns:
            List of workspace information dictionaries
        """
        try:
            workspaces = []

            if project:
                # Get reports for specific project using correct API call
                try:
                    # Use the correct project path format for reports API
                    project_path = f"{entity}/{project}"
                    # Create fresh API client to bypass caching issues
                    fresh_api = self._create_fresh_api_client()
                    reports = fresh_api.reports(project_path)

                    for report in reports:
                        workspace_info = {
                            "name": getattr(report, "display_name", getattr(report, "name", "Untitled")),
                            "id": getattr(report, "id", "unknown"),
                            "url": getattr(report, "url", ""),
                            "entity": entity,
                            "project": project,
                            "created_at": str(report.created_at)
                            if hasattr(report, "created_at") and report.created_at
                            else None,
                            "updated_at": str(report.updated_at)
                            if hasattr(report, "updated_at") and report.updated_at
                            else None,
                            "description": getattr(report, "description", ""),
                        }
                        workspaces.append(workspace_info)

                except Exception as e:
                    logger.warning(f"Could not list reports for {entity}/{project}: {e}")
            else:
                # List reports across all accessible projects
                try:
                    # Note: Standard WandB API doesn't support reports() properly
                    # For now, return empty list - reports need to be created via wandb_workspaces
                    logger.info(f"Listing reports for all projects under {entity} - using fallback")
                    # Could potentially use wandb_workspaces here in the future
                    pass
                except Exception as e:
                    logger.warning(f"Could not list projects for entity {entity}: {e}")

            return workspaces

        except Exception as e:
            logger.error(f"Failed to list workspaces: {e}")
            raise

    async def get_project_metrics(
        self, entity: str, project: str, run_filters: Optional[Dict[str, Any]] = None
    ) -> List[Dict[str, Any]]:
        """Get available metrics for a project.

        Args:
            entity: WandB entity
            project: WandB project name
            run_filters: Optional filters for runs

        Returns:
            List of metric information
        """
        try:
            logger.info(f"Getting metrics for {entity}/{project}")

            # Get runs from the project using the correct API method
            runs = self.api.runs(f"{entity}/{project}")

            # Check if we have any runs
            runs_list = list(runs)
            if not runs_list:
                logger.warning(f"No runs found for {entity}/{project}")
                return []

            logger.info(f"Found {len(runs_list)} runs for {entity}/{project}")

            # Collect unique metrics across all runs
            all_metrics = set()
            metric_info = {}

            for run in runs_list:
                # Apply run filters if provided
                if run_filters and isinstance(run_filters, dict):
                    # Simple filter implementation - can be expanded
                    if "state" in run_filters and hasattr(run, "state") and run.state != run_filters["state"]:
                        continue
                    if "tags" in run_filters and hasattr(run, "tags") and run.tags:
                        run_tags = set(run.tags)
                        filter_tags = set(run_filters["tags"])
                        if not filter_tags.intersection(run_tags):
                            continue

                # Get metrics from run history
                if hasattr(run, "history") and callable(run.history):
                    try:
                        # Get a sample of the history to identify metrics
                        history_sample = run.history(samples=1)
                        if history_sample is not None and not history_sample.empty:
                            for column in history_sample.columns:
                                if column not in ["_step", "_runtime", "_timestamp"]:
                                    all_metrics.add(column)

                                    # Store metric info
                                    if column not in metric_info:
                                        metric_info[column] = {
                                            "name": column,
                                            "type": "numeric",  # Could be inferred from data
                                            "runs_count": 0,
                                            "sample_values": [],
                                        }

                                    metric_info[column]["runs_count"] += 1

                                    # Store sample value
                                    try:
                                        sample_val = history_sample[column].iloc[0]
                                        if len(metric_info[column]["sample_values"]) < 5:
                                            # Convert to JSON-serializable type
                                            serializable_val = self._make_serializable(sample_val)
                                            metric_info[column]["sample_values"].append(serializable_val)
                                    except (IndexError, KeyError):
                                        logger.warning(
                                            f"Could not get sample value for column {column} in run {run.name}"
                                        )
                        else:
                            logger.debug(f"Run {run.name} has empty history")

                    except Exception as e:
                        logger.warning(f"Could not get history for run {run.name}: {e}")
                        continue

                # Get metrics from run summary
                if hasattr(run, "summary") and run.summary is not None:
                    try:
                        # Additional safety check - ensure summary is dict-like
                        if hasattr(run.summary, "items") and callable(run.summary.items):
                            summary_items = run.summary.items()
                            for key, value in summary_items:
                                if not key.startswith("_"):
                                    all_metrics.add(key)

                                    if key not in metric_info:
                                        metric_info[key] = {
                                            "name": key,
                                            "type": "summary",
                                            "runs_count": 0,
                                            "sample_values": [],
                                        }

                                metric_info[key]["runs_count"] += 1
                                if len(metric_info[key]["sample_values"]) < 5:
                                    # Convert to JSON-serializable type
                                    serializable_val = self._make_serializable(value)
                                    metric_info[key]["sample_values"].append(serializable_val)
                        else:
                            logger.warning(f"Run {run.name} summary is not dict-like: {type(run.summary)}")
                    except Exception as e:
                        logger.warning(f"Could not access summary for run {run.name}: {e}")
                        continue

            # Convert to list format
            metrics_list = []
            for metric_name in sorted(all_metrics):
                if metric_name in metric_info:
                    metrics_list.append(metric_info[metric_name])

            logger.info(f"Found {len(metrics_list)} metrics for {entity}/{project}")
            return metrics_list

        except Exception as e:
            logger.error(f"Failed to get project metrics: {e}")
            raise

    async def get_run_details(self, entity: str, project: str, run_id: str) -> Dict[str, Any]:
        """Get detailed information about a specific run.

        Args:
            entity: WandB entity
            project: WandB project name
            run_id: Run ID

        Returns:
            Run details dictionary
        """
        try:
            run = self.api.run(f"{entity}/{project}/{run_id}")

            return {
                "id": run.id,
                "name": run.name,
                "state": run.state,
                "config": dict(run.config),
                "summary": dict(run.summary),
                "tags": run.tags,
                "created_at": run.created_at.isoformat() if run.created_at else None,
                "updated_at": run.updated_at.isoformat() if run.updated_at else None,
                "url": run.url,
            }

        except Exception as e:
            logger.error(f"Failed to get run details: {e}")
            raise

    def is_authenticated(self) -> bool:
        """Check if the client is properly authenticated."""
        try:
            # Try to access the API
            if self.api:
                # Simple check - try to get current user
                self.api.viewer
                return True
            return False
        except Exception as e:
            logger.warning(f"Authentication check failed: {e}")
            return False

    def _make_serializable(self, value):
        """Convert WandB objects to JSON-serializable types."""
        import numpy as np
        import pandas as pd

        # Handle None
        if value is None:
            return None

        # Handle numpy types
        if isinstance(value, np.integer):
            return int(value)
        elif isinstance(value, np.floating):
            return float(value)
        elif isinstance(value, np.ndarray):
            return value.tolist()

        # Handle pandas types
        if isinstance(value, (pd.Series, pd.DataFrame)):
            return value.to_dict()

        # Handle WandB specific types - convert to basic types
        if hasattr(value, "__dict__"):
            # If it has a simple string representation, use that
            try:
                # Try to convert to basic types first
                if hasattr(value, "item"):  # numpy scalars
                    return value.item()
                # For complex objects, try to extract meaningful data
                return str(value)
            except Exception:
                return str(value)

        # Handle standard JSON-serializable types
        if isinstance(value, (str, int, float, bool, list, dict)):
            return value

        # Default: convert to string
        return str(value)

    async def create_dashboard(
        self, entity: str, project: str, name: str, description: str = "", sections: List[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """Create a new WandB dashboard/report."""
        try:
            logger.info(f"Creating dashboard '{name}' for {entity}/{project}")

            # Create report blocks based on sections
            blocks = [wr.H1(name)]

            if description:
                blocks.append(wr.MarkdownBlock(text=description))

            # Add sections as dashboard panels
            if sections:
                for section in sections:
                    section_name = section.get("name", "Untitled Section")
                    blocks.append(wr.H2(section_name))

                    panels = section.get("panels", [])
                    if panels:
                        # Create panels for this section
                        section_panels = []

                        for panel in panels:
                            panel_type = panel.get("type")
                            config = panel.get("config", {})

                            try:
                                # Convert panel configurations to WandB report panels
                                if panel_type == "line_plot":
                                    # Create a line plot panel
                                    line_plot = wr.LinePlot(
                                        title=config.get("title", "Line Plot"),
                                        x=config.get("x_axis", "step"),
                                        y=config.get("y_axis", ["value"]),
                                    )
                                    section_panels.append(line_plot)

                                elif panel_type == "bar_plot":
                                    # Create a bar chart - use ScalarChart as closest equivalent
                                    bar_chart = wr.ScalarChart(
                                        title=config.get("title", "Bar Chart"),
                                        metric=config.get("y_axis", ["value"])[0] if config.get("y_axis") else "value",
                                    )
                                    section_panels.append(bar_chart)

                                elif panel_type == "scatter_plot":
                                    # Create a scatter plot
                                    scatter_plot = wr.ScalarChart(
                                        title=config.get("title", "Scatter Plot"), metric=config.get("y_axis", "value")
                                    )
                                    section_panels.append(scatter_plot)

                                elif panel_type == "scalar_chart":
                                    scalar_chart = wr.ScalarChart(
                                        title=config.get("title", "Scalar Chart"),
                                        metric=config.get("metrics", ["accuracy"])[0]
                                        if config.get("metrics")
                                        else "accuracy",
                                    )
                                    section_panels.append(scalar_chart)

                            except Exception as panel_error:
                                logger.warning(f"Failed to create panel {panel_type}: {panel_error}")
                                continue

                        # Add panels to the section using PanelGrid
                        if section_panels:
                            # Wrap panels in a PanelGrid - this is required by WandB workspace
                            try:
                                panel_grid = wr.PanelGrid(panels=section_panels)
                                blocks.append(panel_grid)
                            except Exception as grid_error:
                                logger.warning(f"Failed to create PanelGrid: {grid_error}")
                                # Fallback: try to add panels directly (might not work but worth trying)
                                blocks.extend(section_panels)

            # Create the report
            report = wr.Report(entity=entity, project=project, title=name, description=description, blocks=blocks)

            # Save the report
            report_url = report.save()

            # Return JSON-serializable data
            result = {
                "status": "success",
                "name": name,
                "entity": entity,
                "project": project,
                "url": str(report_url),  # Convert to string for JSON serialization
                "id": str(report.id) if hasattr(report, "id") else None,
                "created_at": str(report.created_at) if hasattr(report, "created_at") else None,
                "sections_created": len(sections) if sections else 0,
                "total_panels": sum(len(s.get("panels", [])) for s in sections) if sections else 0,
            }

            logger.info(f"Successfully created dashboard: {result}")
            return result

        except Exception as e:
            logger.error(f"Failed to create dashboard: {e}")
            # Return JSON-serializable error info
            return {"status": "error", "error": str(e), "name": name, "entity": entity, "project": project}

    async def update_dashboard(self, dashboard_url: str, modifications: Dict[str, Any]) -> Dict[str, Any]:
        """Update an existing dashboard/report in-place with true modifications."""
        try:
            logger.info(f"Updating dashboard: {dashboard_url}")

            # Load the existing report for true update
            logger.info("Loading existing report for in-place modification...")
            report = wr.Report.from_url(dashboard_url)

            # Store original values for comparison
            original_title = report.title
            original_description = report.description
            original_blocks_count = len(report.blocks) if hasattr(report.blocks, "__len__") else 0

            logger.info(f"Loaded report: '{original_title}' with {original_blocks_count} blocks")

            # Apply property modifications
            if "title" in modifications:
                report.title = modifications["title"]
                logger.info(f"Updated title: '{original_title}' -> '{report.title}'")

            if "description" in modifications:
                report.description = modifications["description"]
                logger.info("Updated description")

            # Apply content modifications
            content_changes = []

            if "add_markdown_section" in modifications:
                markdown_content = modifications["add_markdown_section"]
                new_block = wr.MarkdownBlock(markdown_content)
                report.blocks.append(new_block)
                content_changes.append("Added markdown section")
                logger.info("Added markdown section to dashboard")

            if "remove_section" in modifications:
                section_name = modifications["remove_section"]
                # For now, just log this - full section removal needs more complex logic
                logger.warning(f"Section removal requested: '{section_name}' - requires manual implementation")
                content_changes.append(f"Section removal requested: {section_name} (needs manual implementation)")

            # Save the updated report in-place
            logger.info("Saving updated report in-place...")
            result = report.save(draft=False, clone=False)

            logger.info(f"Successfully updated dashboard in-place: {result.url}")

            # Build response with details of what changed
            changes_made = []
            if original_title != report.title:
                changes_made.append(f"title: '{original_title}' -> '{report.title}'")
            if original_description != report.description:
                changes_made.append("description updated")
            changes_made.extend(content_changes)

            return {
                "status": "success",
                "message": "Dashboard updated successfully in-place",
                "dashboard_url": str(result.url),
                "changes_made": changes_made,
                "applied_modifications": modifications,
                "original_values": {
                    "title": original_title,
                    "description": original_description,
                    "blocks_count": original_blocks_count,
                },
                "current_values": {
                    "title": report.title,
                    "description": report.description,
                    "blocks_count": len(report.blocks) if hasattr(report.blocks, "__len__") else 0,
                },
                "note": "Dashboard was updated in-place - no clone created",
            }

        except Exception as e:
            logger.error(f"Failed to update dashboard: {e}")
            return {"status": "error", "error": str(e), "url": dashboard_url}

    async def clone_dashboard(self, source_url: str, new_name: str) -> Dict[str, Any]:
        """Clone an existing dashboard."""
        try:
            logger.info(f"Cloning dashboard from {source_url} as '{new_name}'")

            # First, get the source dashboard config
            source_config = await self.get_dashboard_config(source_url)

            if source_config.get("status") == "error":
                return {
                    "status": "error",
                    "error": f"Cannot access source dashboard: {source_config.get('error')}",
                    "source_url": source_url,
                    "new_name": new_name,
                }

            entity = source_config["entity"]
            project = source_config["project"]

            # Create a clone with the same entity/project and metadata

            # Create cloned report with source properties
            source_name = source_config.get("name", "Unknown Dashboard")
            source_description = source_config.get("description", "")

            clone_description = f"Clone of: {source_name}"
            if source_description:
                clone_description += f" - {source_description}"

            logger.info(f"Creating dashboard clone: {new_name}")
            cloned_report = wr.Report(
                entity=entity, project=project, title=new_name, description=clone_description, width="readable"
            )

            # Save the cloned report
            result_report = cloned_report.save(clone=True)
            result_url = result_report.url if hasattr(result_report, "url") else str(result_report)

            logger.info(f"Successfully cloned dashboard: {result_url}")

            return {
                "status": "success",
                "message": f"Dashboard successfully cloned as '{new_name}'",
                "source_url": source_url,
                "cloned_url": result_url,
                "clone_info": {
                    "name": new_name,
                    "description": clone_description,
                    "entity": entity,
                    "project": project,
                },
                "source_config": source_config,
            }

        except Exception as e:
            logger.error(f"Failed to clone dashboard: {e}")
            return {"status": "error", "error": str(e), "source_url": source_url, "new_name": new_name}

    async def get_dashboard_config(self, dashboard_url: str) -> Dict[str, Any]:
        """Get configuration of an existing dashboard/report."""
        try:
            logger.info(f"Getting configuration for dashboard: {dashboard_url}")

            # Parse the URL to extract entity, project, and report ID
            # URL format: https://wandb.ai/{entity}/{project}/reports/{report_name}--{report_id}
            if "wandb.ai" not in dashboard_url:
                raise ValueError(f"Invalid WandB URL format: {dashboard_url}")

            # Extract components from URL
            url_parts = dashboard_url.replace("https://wandb.ai/", "").split("/")
            if len(url_parts) < 3 or "reports" not in url_parts:
                raise ValueError(f"Cannot parse dashboard URL: {dashboard_url}")

            entity = url_parts[0]
            project = url_parts[1]

            # Extract report ID from the last part (after the double dash)
            report_part = url_parts[-1]  # e.g., "Report-Name--VmlldzoxNDE4ODY1Ng"
            if "--" not in report_part:
                raise ValueError(f"Cannot extract report ID from URL: {dashboard_url}")

            target_report_id = report_part.split("--")[-1]

            # Find the report using the standard API
            project_path = f"{entity}/{project}"
            # Create fresh API client to bypass caching issues
            fresh_api = self._create_fresh_api_client()
            reports = fresh_api.reports(project_path)

            found_report = None
            for report in reports:
                # Handle both cases: URL may be missing trailing == padding
                if report.id == target_report_id or report.id == target_report_id + "==":
                    found_report = report
                    break

            if not found_report:
                raise ValueError(f"Report not found: {target_report_id}")

            # Return report configuration
            config = {
                "id": found_report.id,
                "name": getattr(found_report, "display_name", getattr(found_report, "name", "Untitled")),
                "url": found_report.url,
                "entity": entity,
                "project": project,
                "description": getattr(found_report, "description", ""),
                "created_at": str(found_report.created_at)
                if hasattr(found_report, "created_at") and found_report.created_at
                else None,
                "updated_at": str(found_report.updated_at)
                if hasattr(found_report, "updated_at") and found_report.updated_at
                else None,
                "status": "success",
            }

            # Try to get sections if available (may have limitations)
            try:
                sections = getattr(found_report, "sections", None)
                if sections:
                    config["sections"] = sections
            except Exception as e:
                logger.warning(f"Could not get sections for report {target_report_id}: {e}")
                config["sections"] = "unavailable"

            return config

        except Exception as e:
            logger.error(f"Failed to get dashboard config: {e}")
            return {"status": "error", "error": str(e), "url": dashboard_url}

    async def add_panel_to_dashboard(
        self, dashboard_url: str, section_name: str, panel_type: str, panel_config: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Add a panel to an existing dashboard."""
        try:
            logger.info(f"Adding {panel_type} panel to '{section_name}' in dashboard: {dashboard_url}")

            # Get current dashboard config
            current_config = await self.get_dashboard_config(dashboard_url)

            if current_config.get("status") == "error":
                return current_config

            # Load the existing report for modification
            logger.info("Loading existing report for panel addition...")
            report = wr.Report.from_url(dashboard_url)

            original_blocks_count = len(report.blocks) if hasattr(report.blocks, "__len__") else 0
            logger.info(f"Loaded report: '{report.title}' with {original_blocks_count} blocks")

            # Create the appropriate block/panel based on panel_type
            new_blocks_added = []

            if panel_type == "line_plot":
                # Add a section header and basic visualization
                section_block = wr.H2(f"📊 {section_name}")
                report.blocks.append(section_block)
                new_blocks_added.append("H2 section header")

                # Add markdown explanation for the panel
                panel_description = (
                    f"**{panel_config.get('title', 'Line Plot')}**\n\nPanel configuration: {str(panel_config)}"
                )
                markdown_block = wr.MarkdownBlock(panel_description)
                report.blocks.append(markdown_block)
                new_blocks_added.append("Line plot description")

            elif panel_type == "markdown" or panel_type == "text":
                # Add markdown content
                content = panel_config.get("content", f"## {section_name}\n\nPanel added via MCP server")
                markdown_block = wr.MarkdownBlock(content)
                report.blocks.append(markdown_block)
                new_blocks_added.append("Markdown panel")

            else:
                # Generic panel - add as markdown with configuration info
                section_block = wr.H2(f"📊 {section_name}")
                report.blocks.append(section_block)

                config_text = f"**{panel_type.title()} Panel**\n\nConfiguration:\n```json\n{str(panel_config)}\n```"
                markdown_block = wr.MarkdownBlock(config_text)
                report.blocks.append(markdown_block)
                new_blocks_added.append(f"{panel_type} panel (as markdown)")

            # Save the updated report
            logger.info(f"Saving report with {len(new_blocks_added)} new blocks...")
            result = report.save(draft=False, clone=False)

            new_blocks_count = len(report.blocks) if hasattr(report.blocks, "__len__") else 0
            logger.info(f"Successfully added panel to dashboard: {result.url}")

            return {
                "status": "success",
                "message": f"Panel added successfully to section '{section_name}'",
                "dashboard_url": str(result.url),
                "panel_added": {
                    "section_name": section_name,
                    "panel_type": panel_type,
                    "blocks_added": new_blocks_added,
                },
                "blocks_before": original_blocks_count,
                "blocks_after": new_blocks_count,
                "blocks_added_count": len(new_blocks_added),
                "note": "Panel added to existing dashboard - no clone created",
            }

        except Exception as e:
            logger.error(f"Failed to add panel to dashboard: {e}")
            return {"status": "error", "error": str(e), "url": dashboard_url}

    async def update_panel(self, dashboard_url: str, panel_identifier: dict, new_content: str) -> Dict[str, Any]:
        """Update content of an existing panel/block in a dashboard."""
        try:
            logger.info(f"Updating panel in dashboard: {dashboard_url}")

            # Load the existing report
            logger.info("Loading existing report for panel update...")
            report = wr.Report.from_url(dashboard_url)

            original_blocks_count = len(report.blocks) if hasattr(report.blocks, "__len__") else 0
            logger.info(f"Loaded report: '{report.title}' with {original_blocks_count} blocks")

            # Find the target panel/block
            target_index = None
            target_block = None
            original_content = None

            # Method 1: Update by index
            if "index" in panel_identifier:
                index = panel_identifier["index"]
                if 0 <= index < len(report.blocks):
                    target_index = index
                    target_block = report.blocks[index]
                    original_content = target_block.text
                    logger.info(f"Found panel by index {index}: {type(target_block).__name__}")
                else:
                    return {
                        "status": "error",
                        "error": f"Panel index {index} out of range (0-{len(report.blocks) - 1})",
                        "url": dashboard_url,
                    }

            # Method 2: Update by content search
            elif "search_text" in panel_identifier:
                search_text = panel_identifier["search_text"]
                for i, block in enumerate(report.blocks):
                    if hasattr(block, "text") and search_text in block.text:
                        target_index = i
                        target_block = block
                        original_content = block.text
                        logger.info(f"Found panel by search '{search_text}' at index {i}: {type(block).__name__}")
                        break

                if target_block is None:
                    return {
                        "status": "error",
                        "error": f"No panel found containing text: '{search_text}'",
                        "url": dashboard_url,
                    }

            # Method 3: Update by block type
            elif "block_type" in panel_identifier:
                block_type = panel_identifier["block_type"].lower()
                type_map = {"markdown": "MarkdownBlock", "h1": "H1", "h2": "H2", "h3": "H3"}

                target_type = type_map.get(block_type, block_type)
                occurrence = panel_identifier.get("occurrence", 0)  # Which occurrence of this type

                matching_blocks = []
                for i, block in enumerate(report.blocks):
                    if type(block).__name__ == target_type:
                        matching_blocks.append((i, block))

                if occurrence < len(matching_blocks):
                    target_index, target_block = matching_blocks[occurrence]
                    original_content = target_block.text
                    logger.info(f"Found {target_type} block #{occurrence} at index {target_index}")
                else:
                    return {
                        "status": "error",
                        "error": f"No {target_type} block found at occurrence #{occurrence}",
                        "url": dashboard_url,
                    }

            else:
                return {
                    "status": "error",
                    "error": "Panel identifier must include 'index', 'search_text', or 'block_type'",
                    "url": dashboard_url,
                }

            # Update the panel content
            logger.info(f"Updating block at index {target_index}")
            logger.info(f"Original content: {repr(original_content[:100])}")

            target_block.text = new_content
            logger.info(f"New content: {repr(new_content[:100])}")

            # Save the updated report
            logger.info("Saving updated report with modified panel...")
            result = report.save(draft=False, clone=False)

            logger.info(f"Successfully updated panel in dashboard: {result.url}")

            return {
                "status": "success",
                "message": "Panel updated successfully",
                "dashboard_url": str(result.url),
                "updated_panel": {
                    "index": target_index,
                    "type": type(target_block).__name__,
                    "identifier": panel_identifier,
                },
                "content_change": {
                    "original": original_content,
                    "updated": new_content,
                    "length_change": len(new_content) - len(original_content),
                },
                "note": "Panel content updated in-place - no clone created",
            }

        except Exception as e:
            logger.error(f"Failed to update panel: {e}")
            return {"status": "error", "error": str(e), "url": dashboard_url}

    async def remove_panel(self, dashboard_url: str, panel_identifier: dict) -> Dict[str, Any]:
        """Remove an existing panel/block from a dashboard."""
        try:
            logger.info(f"Removing panel from dashboard: {dashboard_url}")

            # Load the existing report
            logger.info("Loading existing report for panel removal...")
            report = wr.Report.from_url(dashboard_url)

            original_blocks_count = len(report.blocks) if hasattr(report.blocks, "__len__") else 0
            logger.info(f"Loaded report: '{report.title}' with {original_blocks_count} blocks")

            # Find the target panel/block to remove
            target_index = None
            target_block = None
            original_content = None

            # Method 1: Remove by index
            if "index" in panel_identifier:
                index = panel_identifier["index"]
                if 0 <= index < len(report.blocks):
                    target_index = index
                    target_block = report.blocks[index]
                    original_content = target_block.text
                    logger.info(f"Found panel to remove by index {index}: {type(target_block).__name__}")
                else:
                    return {
                        "status": "error",
                        "error": f"Panel index {index} out of range (0-{len(report.blocks) - 1})",
                        "url": dashboard_url,
                    }

            # Method 2: Remove by content search
            elif "search_text" in panel_identifier:
                search_text = panel_identifier["search_text"]
                for i, block in enumerate(report.blocks):
                    if hasattr(block, "text") and search_text in block.text:
                        target_index = i
                        target_block = block
                        original_content = block.text
                        logger.info(
                            f"Found panel to remove by search '{search_text}' at index {i}: {type(block).__name__}"
                        )
                        break

                if target_block is None:
                    return {
                        "status": "error",
                        "error": f"No panel found containing text: '{search_text}'",
                        "url": dashboard_url,
                    }

            # Method 3: Remove by block type
            elif "block_type" in panel_identifier:
                block_type = panel_identifier["block_type"].lower()
                type_map = {"markdown": "MarkdownBlock", "h1": "H1", "h2": "H2", "h3": "H3"}

                target_type = type_map.get(block_type, block_type)
                occurrence = panel_identifier.get("occurrence", 0)  # Which occurrence of this type

                matching_blocks = []
                for i, block in enumerate(report.blocks):
                    if type(block).__name__ == target_type:
                        matching_blocks.append((i, block))

                if occurrence < len(matching_blocks):
                    target_index, target_block = matching_blocks[occurrence]
                    original_content = target_block.text
                    logger.info(f"Found {target_type} block #{occurrence} to remove at index {target_index}")
                else:
                    return {
                        "status": "error",
                        "error": f"No {target_type} block found at occurrence #{occurrence}",
                        "url": dashboard_url,
                    }

            else:
                return {
                    "status": "error",
                    "error": "Panel identifier must include 'index', 'search_text', or 'block_type'",
                    "url": dashboard_url,
                }

            # Remove the panel
            logger.info(f"Removing block at index {target_index}")
            logger.info(f"Removing content: {repr(original_content[:100])}")

            removed_block = report.blocks.pop(target_index)

            # Save the updated report
            logger.info("Saving report after panel removal...")
            result = report.save(draft=False, clone=False)

            new_blocks_count = len(report.blocks) if hasattr(report.blocks, "__len__") else 0
            logger.info(f"Successfully removed panel from dashboard: {result.url}")

            return {
                "status": "success",
                "message": "Panel removed successfully",
                "dashboard_url": str(result.url),
                "removed_panel": {
                    "original_index": target_index,
                    "type": type(removed_block).__name__,
                    "identifier": panel_identifier,
                    "content": original_content,
                },
                "blocks_before": original_blocks_count,
                "blocks_after": new_blocks_count,
                "blocks_removed_count": 1,
                "note": "Panel removed from dashboard - remaining panels reindexed automatically",
            }

        except Exception as e:
            logger.error(f"Failed to remove panel: {e}")
            return {"status": "error", "error": str(e), "url": dashboard_url}

    async def create_custom_chart(
        self, entity: str, project: str, metrics: List[str], chart_type: str, config: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Create a custom chart/visualization with specified metrics and configuration."""
        try:
            logger.info(f"Creating custom {chart_type} chart for {entity}/{project} with metrics: {metrics}")

            # Create a new report to hold our custom chart
            chart_title = config.get("title", f"Custom {chart_type.title()} Chart")
            report_title = config.get("report_title", f"{chart_title} - Dashboard")
            report_description = config.get("report_description", f"Custom {chart_type} visualization created via MCP")

            report = wr.Report(entity=entity, project=project, title=report_title, description=report_description)

            # Add a header for the chart
            header = wr.H1(chart_title)
            report.blocks.append(header)

            # Create the appropriate chart based on chart_type
            chart_block = None

            if chart_type.lower() == "line_plot" or chart_type.lower() == "lineplot":
                logger.info("Creating LinePlot chart...")
                chart_block = wr.LinePlot(
                    x=config.get("x_axis", "step"),
                    y=metrics,
                    title=chart_title,
                    title_x=config.get("x_label", "Steps"),
                    title_y=config.get("y_label", "Value"),
                    smoothing_factor=config.get("smoothing", None),
                    log_x=config.get("log_x", None),
                    log_y=config.get("log_y", None),
                    max_runs_to_show=config.get("max_runs", None),
                    groupby=config.get("groupby", None),  # Use None instead of []
                    legend_position=config.get(
                        "legend_position", None
                    ),  # Use None, valid options: 'north', 'south', 'east', 'west'
                )

            elif chart_type.lower() == "bar_plot" or chart_type.lower() == "barplot":
                logger.info("Creating BarPlot chart...")
                # Map orientation values: 'vertical' -> 'v', 'horizontal' -> 'h'
                orientation = config.get("orientation", "vertical")
                if orientation == "vertical":
                    orientation = "v"
                elif orientation == "horizontal":
                    orientation = "h"

                chart_block = wr.BarPlot(
                    metrics=metrics,
                    title=chart_title,
                    title_x=config.get("x_label", "Metric"),
                    title_y=config.get("y_label", "Value"),
                    orientation=orientation,  # Use 'v' or 'h'
                    max_bars_to_show=config.get("max_bars", None),
                    max_runs_to_show=config.get("max_runs", None),
                    groupby=config.get("groupby", None),  # Use None instead of []
                )

            elif chart_type.lower() == "scatter_plot" or chart_type.lower() == "scatterplot":
                logger.info("Creating ScatterPlot chart...")
                x_metric = metrics[0] if len(metrics) > 0 else config.get("x_metric", "step")
                y_metric = metrics[1] if len(metrics) > 1 else metrics[0] if metrics else config.get("y_metric", "loss")
                z_metric = metrics[2] if len(metrics) > 2 else config.get("z_metric", None)

                chart_block = wr.ScatterPlot(
                    x=x_metric,
                    y=y_metric,
                    z=z_metric,
                    title=chart_title,
                    log_x=config.get("log_x", None),
                    log_y=config.get("log_y", None),
                    log_z=config.get("log_z", None),
                    regression=config.get("show_regression", None),
                    gradient=config.get("gradient", None),
                )

            elif chart_type.lower() == "custom_chart" or chart_type.lower() == "customchart":
                logger.info("Creating CustomChart...")
                # CustomChart is more advanced - requires query configuration
                query_config = config.get("query", {})
                chart_fields = config.get("chart_fields", {})
                chart_strings = config.get("chart_strings", {})

                chart_block = wr.CustomChart(
                    chart_name=chart_title,
                    query=query_config,
                    chart_fields=chart_fields,
                    chart_strings=chart_strings,
                )

            else:
                return {
                    "status": "error",
                    "error": f"Unsupported chart type: {chart_type}. "
                    f"Supported types: line_plot, bar_plot, scatter_plot, custom_chart",
                }

            # Wrap the chart in a PanelGrid and add to report
            if chart_block:
                # PanelGrid expects a flat list of panel objects, not nested lists
                panel_grid = wr.PanelGrid(panels=[chart_block])
                report.blocks.append(panel_grid)

                # Add a description section
                description_text = f"""
## Chart Configuration

**Chart Type:** {chart_type}
**Metrics:** {", ".join(metrics)}
**Entity/Project:** {entity}/{project}

**Configuration:**
```json
{json.dumps(config, indent=2)}
```

This chart was created using the WandB MCP server's `create_custom_chart` tool.
"""
                description_block = wr.MarkdownBlock(description_text)
                report.blocks.append(description_block)

                # Save the report
                logger.info("Saving custom chart dashboard...")
                result = report.save()

                logger.info(f"Custom chart created successfully: {result.url}")

                return {
                    "status": "success",
                    "message": f"Custom {chart_type} chart created successfully",
                    "dashboard_url": str(result.url),
                    "chart_config": {
                        "title": chart_title,
                        "type": chart_type,
                        "metrics": metrics,
                        "entity": entity,
                        "project": project,
                        "configuration": config,
                    },
                    "note": "Chart created as a new dashboard - "
                    "use update_panel or add_panel to modify existing dashboards",
                }

            else:
                return {"status": "error", "error": f"Failed to create {chart_type} chart"}

        except Exception as e:
            logger.error(f"Failed to create custom chart: {e}")
            return {"status": "error", "error": str(e)}

    async def bulk_delete_dashboards(self, dashboard_urls: List[str], confirmed: bool = False) -> Dict[str, Any]:
        """Bulk delete multiple WandB dashboards with confirmation.

        Args:
            dashboard_urls: List of dashboard URLs to delete
            confirmed: Whether the user has confirmed the deletion

        Returns:
            dict: Result of the bulk deletion operation
        """
        logger.info(f"Bulk delete request for {len(dashboard_urls)} dashboards, confirmed={confirmed}")

        if not dashboard_urls:
            return {"status": "error", "error": "No dashboard URLs provided"}

        # First pass: Get details of all dashboards for confirmation
        dashboard_details = []
        invalid_urls = []

        for url in dashboard_urls:
            try:
                report = wr.Report.from_url(url)
                dashboard_details.append(
                    {
                        "url": url,
                        "title": report.title,
                        "id": report.id,
                        "report_object": report,
                    }
                )
            except Exception as e:
                logger.warning(f"Failed to load dashboard {url}: {e}")
                invalid_urls.append({"url": url, "error": str(e)})

        # If not confirmed, return confirmation request with details
        if not confirmed:
            return {
                "status": "confirmation_required",
                "message": "Bulk deletion requires confirmation",
                "dashboards_to_delete": [
                    {
                        "url": d["url"],
                        "title": d["title"],
                        "id": d["id"],
                    }
                    for d in dashboard_details
                ],
                "invalid_urls": invalid_urls,
                "total_valid": len(dashboard_details),
                "total_invalid": len(invalid_urls),
                "confirmation_message": f"Are you sure you want to delete {len(dashboard_details)} dashboards? "
                f"This action cannot be undone.",
                "next_step": "Call this function again with confirmed=True to proceed with deletion",
            }

        # Confirmed deletion: Proceed with actual deletion
        successful_deletions = []
        failed_deletions = []

        for dashboard in dashboard_details:
            try:
                logger.info(f"Deleting dashboard: {dashboard['title']} ({dashboard['id']})")
                delete_result = dashboard["report_object"].delete()

                if delete_result:
                    successful_deletions.append(
                        {
                            "url": dashboard["url"],
                            "title": dashboard["title"],
                            "id": dashboard["id"],
                            "status": "deleted",
                        }
                    )
                else:
                    failed_deletions.append(
                        {
                            "url": dashboard["url"],
                            "title": dashboard["title"],
                            "id": dashboard["id"],
                            "error": "Delete operation returned False",
                        }
                    )
            except Exception as e:
                logger.error(f"Failed to delete dashboard {dashboard['title']}: {e}")
                failed_deletions.append(
                    {
                        "url": dashboard["url"],
                        "title": dashboard["title"],
                        "id": dashboard["id"],
                        "error": str(e),
                    }
                )

        # Return comprehensive results
        return {
            "status": "completed",
            "message": f"Bulk deletion completed: {len(successful_deletions)} successful, "
            f"{len(failed_deletions)} failed",
            "successful_deletions": successful_deletions,
            "failed_deletions": failed_deletions,
            "invalid_urls": invalid_urls,
            "summary": {
                "total_requested": len(dashboard_urls),
                "valid_dashboards": len(dashboard_details),
                "successful": len(successful_deletions),
                "failed": len(failed_deletions),
                "invalid": len(invalid_urls),
            },
        }

    async def _delete_single_dashboard(self, dashboard_url: str) -> dict:
        """Delete an existing WandB dashboard/report.

        Args:
            dashboard_url: URL of the dashboard to delete

        Returns:
            dict: Result of the deletion operation
        """
        logger.info(f"Attempting to delete dashboard: {dashboard_url}")

        try:
            # Use wandb_workspaces to load the report directly from URL
            # This gives us a Report object with the delete() method
            report = wr.Report.from_url(dashboard_url)

            # Get report details before deletion
            report_id = report.id
            report_title = report.title

            logger.info(f"Deleting report: {report_title} (ID: {report_id})")

            # Delete the report
            delete_result = report.delete()

            if delete_result:
                return {
                    "status": "success",
                    "message": f"Dashboard '{report_title}' deleted successfully",
                    "deleted_report_id": report_id,
                    "deleted_report_title": report_title,
                    "url": dashboard_url,
                }
            else:
                return {
                    "status": "error",
                    "message": f"Failed to delete dashboard '{report_title}' - delete() returned False",
                    "report_id": report_id,
                    "report_title": report_title,
                    "url": dashboard_url,
                }

        except Exception as e:
            logger.error(f"Error deleting dashboard: {e}")
            return {
                "status": "error",
                "message": f"Error deleting dashboard: {str(e)}",
                "url": dashboard_url,
            }
