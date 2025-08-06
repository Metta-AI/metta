"""
Metadata management utilities for analysis workflows.
"""

import json
from dataclasses import asdict, dataclass
from datetime import datetime
from enum import Enum
from pathlib import Path
from typing import Any, Dict, List, Optional


class AnalysisType(Enum):
    """Types of analysis that can be performed."""

    ACTIVATION_RECORDING = "activation_recording"
    SAE_TRAINING = "sae_training"
    CONCEPT_ANALYSIS = "concept_analysis"
    CONCEPT_STEERING = "concept_steering"
    CROSS_POLICY = "cross_policy"


@dataclass
class AnalysisMetadata:
    """Metadata for an analysis run."""

    analysis_type: AnalysisType
    policy_uri: str
    environment: str
    timestamp: str
    description: str
    parameters: Dict[str, Any]
    input_files: List[str]
    output_files: List[str]
    status: str = "running"  # running, completed, failed
    error_message: Optional[str] = None


class MetadataManager:
    """
    Manages metadata for analysis workflows.
    """

    def __init__(self, metadata_dir: str = "metadata"):
        """
        Initialize the metadata manager.

        Args:
            metadata_dir: Directory to store metadata
        """
        self.metadata_dir = Path(metadata_dir)
        self.metadata_dir.mkdir(exist_ok=True)

    def create_analysis_metadata(
        self,
        analysis_type: AnalysisType,
        policy_uri: str,
        environment: str,
        description: str,
        parameters: Dict[str, Any],
        input_files: List[str] = None,
    ) -> AnalysisMetadata:
        """
        Create metadata for a new analysis run.

        Args:
            analysis_type: Type of analysis
            policy_uri: Wandb URI of the policy
            environment: Environment name
            description: Description of the analysis
            parameters: Analysis parameters
            input_files: List of input file paths

        Returns:
            Analysis metadata
        """
        metadata = AnalysisMetadata(
            analysis_type=analysis_type,
            policy_uri=policy_uri,
            environment=environment,
            timestamp=datetime.now().isoformat(),
            description=description,
            parameters=parameters,
            input_files=input_files or [],
            output_files=[],
            status="running",
        )

        return metadata

    def save_metadata(self, metadata: AnalysisMetadata) -> Path:
        """
        Save analysis metadata to file.

        Args:
            metadata: Analysis metadata

        Returns:
            Path to saved metadata file
        """
        # Generate filename
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        policy_name = metadata.policy_uri.replace("/", "_")
        analysis_name = metadata.analysis_type.value
        filename = f"{analysis_name}_{policy_name}_{timestamp}.json"
        filepath = self.metadata_dir / filename

        # Convert to dict and save
        metadata_dict = asdict(metadata)
        metadata_dict["analysis_type"] = metadata.analysis_type.value

        with open(filepath, "w") as f:
            json.dump(metadata_dict, f, indent=2)

        return filepath

    def load_metadata(self, filepath: Path) -> AnalysisMetadata:
        """
        Load analysis metadata from file.

        Args:
            filepath: Path to metadata file

        Returns:
            Loaded analysis metadata
        """
        with open(filepath, "r") as f:
            data = json.load(f)

        # Convert back to AnalysisMetadata
        data["analysis_type"] = AnalysisType(data["analysis_type"])
        return AnalysisMetadata(**data)

    def update_metadata_status(
        self, filepath: Path, status: str, output_files: List[str] = None, error_message: Optional[str] = None
    ):
        """
        Update the status of an analysis run.

        Args:
            filepath: Path to metadata file
            status: New status
            output_files: List of output files
            error_message: Error message if failed
        """
        metadata = self.load_metadata(filepath)
        metadata.status = status

        if output_files:
            metadata.output_files = output_files

        if error_message:
            metadata.error_message = error_message

        # Save updated metadata
        self.save_metadata(metadata)

    def list_analyses(
        self,
        analysis_type: Optional[AnalysisType] = None,
        policy_uri: Optional[str] = None,
        status: Optional[str] = None,
    ) -> List[AnalysisMetadata]:
        """
        List analysis runs with optional filtering.

        Args:
            analysis_type: Filter by analysis type
            policy_uri: Filter by policy URI
            status: Filter by status

        Returns:
            List of matching analysis metadata
        """
        analyses = []

        for filepath in self.metadata_dir.glob("*.json"):
            try:
                metadata = self.load_metadata(filepath)

                # Apply filters
                if analysis_type and metadata.analysis_type != analysis_type:
                    continue
                if policy_uri and metadata.policy_uri != policy_uri:
                    continue
                if status and metadata.status != status:
                    continue

                analyses.append(metadata)

            except Exception as e:
                print(f"Error loading metadata from {filepath}: {e}")

        # Sort by timestamp (newest first)
        analyses.sort(key=lambda x: x.timestamp, reverse=True)

        return analyses

    def get_analysis_summary(self) -> Dict[str, Any]:
        """
        Get summary of all analyses.

        Returns:
            Summary statistics
        """
        analyses = self.list_analyses()

        summary = {
            "total_analyses": len(analyses),
            "by_type": {},
            "by_status": {},
            "by_policy": {},
            "recent_analyses": [],
        }

        for analysis in analyses:
            # Count by type
            type_name = analysis.analysis_type.value
            summary["by_type"][type_name] = summary["by_type"].get(type_name, 0) + 1

            # Count by status
            summary["by_status"][analysis.status] = summary["by_status"].get(analysis.status, 0) + 1

            # Count by policy
            summary["by_policy"][analysis.policy_uri] = summary["by_policy"].get(analysis.policy_uri, 0) + 1

            # Recent analyses (last 10)
            if len(summary["recent_analyses"]) < 10:
                summary["recent_analyses"].append(
                    {
                        "type": type_name,
                        "policy": analysis.policy_uri,
                        "status": analysis.status,
                        "timestamp": analysis.timestamp,
                    }
                )

        return summary

    def create_workflow_metadata(self, workflow_name: str, steps: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Create metadata for a multi-step workflow.

        Args:
            workflow_name: Name of the workflow
            steps: List of workflow steps

        Returns:
            Workflow metadata
        """
        workflow_metadata = {
            "workflow_name": workflow_name,
            "created_at": datetime.now().isoformat(),
            "steps": steps,
            "current_step": 0,
            "status": "pending",
            "results": {},
        }

        return workflow_metadata

    def update_workflow_progress(self, workflow_metadata: Dict[str, Any], step_index: int, step_result: Dict[str, Any]):
        """
        Update workflow progress.

        Args:
            workflow_metadata: Workflow metadata
            step_index: Index of completed step
            step_result: Results from the step
        """
        workflow_metadata["current_step"] = step_index
        workflow_metadata["results"][f"step_{step_index}"] = step_result

        if step_index >= len(workflow_metadata["steps"]) - 1:
            workflow_metadata["status"] = "completed"
        else:
            workflow_metadata["status"] = "running"
