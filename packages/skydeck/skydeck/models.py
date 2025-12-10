"""Data models for SkyDeck dashboard."""

from datetime import datetime
from enum import Enum
from typing import Optional, Union

from pydantic import BaseModel, Field


class JobStatus(str, Enum):
    """Status of a single job execution."""

    INIT = "INIT"  # Job created but not yet submitted
    PENDING = "PENDING"  # Submitted to SkyPilot, waiting to start
    STARTING = "STARTING"  # Job is starting (provisioning resources)
    RUNNING = "RUNNING"  # Currently executing
    SUCCEEDED = "SUCCEEDED"  # Completed successfully
    FAILED = "FAILED"  # Completed with error
    CANCELLED = "CANCELLED"  # User cancelled
    UNKNOWN = "UNKNOWN"  # Cannot determine status


class DesiredState(str, Enum):
    """Desired state of an experiment."""

    RUNNING = "RUNNING"  # Should be running
    STOPPED = "STOPPED"  # Should be stopped (cluster exists but idle)
    TERMINATED = "TERMINATED"  # Should be terminated (no cluster)


class Job(BaseModel):
    """Single SkyPilot job execution.

    Represents one execution of an experiment. Multiple jobs can exist
    for a single experiment (e.g., after crashes/restarts).
    """

    # Identification
    id: str = Field(..., description="Unique job ID: {experiment_id}-{job_number}")
    experiment_id: str = Field(..., description="Parent experiment ID")
    cluster_name: str = Field(..., description="SkyPilot cluster name")
    sky_job_id: Optional[int] = Field(None, description="SkyPilot's internal job ID")

    # Status
    status: JobStatus = Field(JobStatus.INIT, description="Current job status")

    # Timestamps
    created_at: datetime = Field(default_factory=datetime.utcnow, description="When job was created")
    submitted_at: Optional[datetime] = Field(None, description="When submitted to SkyPilot")
    started_at: Optional[datetime] = Field(None, description="When job started running")
    ended_at: Optional[datetime] = Field(None, description="When job finished")

    # Metadata
    command: str = Field(..., description="Full command executed")
    logs_path: Optional[str] = Field(None, description="Path to job logs")
    exit_code: Optional[int] = Field(None, description="Exit code if finished")
    error_message: Optional[str] = Field(None, description="Error message if failed")

    # Resources
    nodes: int = Field(1, description="Number of nodes")
    gpus: int = Field(0, description="Number of GPUs per node")
    instance_type: Optional[str] = Field(None, description="Instance type")
    cloud: Optional[str] = Field(None, description="Cloud provider")

    def is_terminal(self) -> bool:
        """Check if job is in a terminal state."""
        return self.status in {JobStatus.SUCCEEDED, JobStatus.FAILED, JobStatus.CANCELLED}

    def is_active(self) -> bool:
        """Check if job is actively running."""
        return self.status in {JobStatus.PENDING, JobStatus.STARTING, JobStatus.RUNNING}


class OperationType(str, Enum):
    """Type of operation performed."""

    START = "START"  # Started an experiment
    STOP = "STOP"  # Stopped an experiment
    CANCEL = "CANCEL"  # Canceled a job
    DELETE = "DELETE"  # Deleted an experiment
    CREATE = "CREATE"  # Created an experiment


class OperationLog(BaseModel):
    """Log entry for operations performed on experiments/jobs."""

    id: Optional[int] = Field(None, description="Auto-generated log ID")
    timestamp: datetime = Field(..., description="When the operation occurred")
    operation_type: OperationType = Field(..., description="Type of operation")
    experiment_id: Optional[int] = Field(None, description="Experiment ID (if applicable)")
    experiment_name: Optional[str] = Field(None, description="Experiment name for display")
    job_id: Optional[str] = Field(None, description="Job ID (if applicable)")
    success: bool = Field(True, description="Whether the operation succeeded")
    error_message: Optional[str] = Field(None, description="Error message if failed")
    output: Optional[str] = Field(None, description="Command output (stdout/stderr)")
    user: Optional[str] = Field(None, description="User who performed the operation")


class Experiment(BaseModel):
    """Configuration template that spawns jobs.

    An experiment defines what should run, and can spawn multiple jobs
    over time (e.g., after failures/restarts). Only one job per experiment
    can be running at a time.
    """

    # Identification
    id: Optional[int] = Field(None, description="Auto-generated experiment ID")
    name: str = Field(..., description="Unique experiment name")

    # State management
    desired_state: DesiredState = Field(DesiredState.STOPPED, description="What state user wants")
    current_state: JobStatus = Field(JobStatus.INIT, description="Actual current status")

    # Configuration as flags
    flags: dict[str, Union[str, int, float, bool]] = Field(default_factory=dict, description="Configuration flags")
    base_command: str = Field("lt", description="Base command (before flags)")
    tool_path: Optional[str] = Field(None, description="Tool path (e.g., recipes.experiment.cog_arena.train)")
    git_branch: Optional[str] = Field(None, description="Git branch name")

    # Job management
    cluster_name: Optional[str] = Field(None, description="SkyPilot cluster name")
    latest_epoch: Optional[int] = Field(None, description="Latest checkpoint epoch")

    # Resource defaults
    nodes: int = Field(1, description="Number of nodes")
    gpus: int = Field(0, description="Number of GPUs per node")
    instance_type: Optional[str] = Field(None, description="Instance type")
    cloud: Optional[str] = Field(None, description="Cloud provider (aws, gcp, azure)")
    spot: bool = Field(False, description="Use spot instances")

    # Metadata
    created_at: datetime = Field(default_factory=datetime.utcnow, description="When experiment was created")
    updated_at: datetime = Field(default_factory=datetime.utcnow, description="Last update time")
    wandb_link: Optional[str] = Field(None, description="W&B run URL")
    description: Optional[str] = Field(None, description="Human description")
    tags: list[str] = Field(default_factory=list, description="Tags for organization")

    # Organization
    group: Optional[str] = Field(None, description="Group name for organizing experiments")
    order: int = Field(0, description="Display order within group")

    # UI State
    is_expanded: bool = Field(False, description="Whether row is expanded in UI")
    starred: bool = Field(False, description="Whether experiment is starred by user")
    deleted: bool = Field(False, description="Whether experiment is soft-deleted")

    def build_command(self) -> str:
        """Construct full command from base_command, gpus, nodes, tool_path, and flags.

        Returns:
            Command string like: "lt --gpus=4 --nodes=4 --git-ref=branch recipe key=val"
        """
        parts = [self.base_command]

        # Always add GPUs and nodes as CLI flags
        parts.append(f"--gpus={self.gpus}")
        parts.append(f"--nodes={self.nodes}")

        # Add git branch if specified (skip invalid values like "-")
        if self.git_branch and self.git_branch != "-":
            parts.append(f"--git-ref={self.git_branch}")

        # Add tool path
        if self.tool_path:
            parts.append(self.tool_path)

        # Always add run name using experiment name
        parts.append(f"run={self.name}")

        # Add all flags
        for key, value in sorted(self.flags.items()):
            if isinstance(value, bool):
                # Boolean flags: key=true or key=false
                parts.append(f"{key}={str(value).lower()}")
            elif isinstance(value, str):
                # String flags: key=value (quote if contains spaces)
                if " " in value:
                    parts.append(f'{key}="{value}"')
                else:
                    parts.append(f"{key}={value}")
            else:
                # Numeric flags: key=value
                parts.append(f"{key}={value}")

        return " ".join(parts)

    def needs_reconciliation(self) -> bool:
        """Check if current state doesn't match desired state."""
        if self.desired_state == DesiredState.RUNNING:
            return self.current_state not in {JobStatus.RUNNING, JobStatus.PENDING}
        elif self.desired_state == DesiredState.STOPPED:
            return self.current_state in {JobStatus.RUNNING, JobStatus.PENDING}
        elif self.desired_state == DesiredState.TERMINATED:
            return self.current_state != JobStatus.INIT
        return False


class Cluster(BaseModel):
    """SkyPilot cluster information."""

    name: str = Field(..., description="Cluster name")
    status: str = Field(..., description="Cluster status (UP, STOPPED, etc)")
    num_nodes: int = Field(0, description="Number of nodes")
    instance_type: Optional[str] = Field(None, description="Instance type")
    cloud: Optional[str] = Field(None, description="Cloud provider")
    created_at: Optional[datetime] = Field(None, description="Creation time")
    last_seen: datetime = Field(default_factory=datetime.utcnow, description="Last time cluster was seen")


# API Request/Response Models


class CreateExperimentRequest(BaseModel):
    """Request to create a new experiment."""

    name: str  # User-facing name, must be unique among non-deleted experiments
    flags: dict[str, Union[str, int, float, bool]] = Field(default_factory=dict)
    base_command: str = "lt"
    tool_path: Optional[str] = None
    git_branch: Optional[str] = None
    nodes: int = 1
    gpus: int = 0
    instance_type: Optional[str] = None
    cloud: Optional[str] = None
    spot: bool = False
    desired_state: DesiredState = DesiredState.STOPPED
    description: Optional[str] = None
    tags: list[str] = Field(default_factory=list)
    group: Optional[str] = None
    order: int = 0


class UpdateDesiredStateRequest(BaseModel):
    """Request to update experiment desired state."""

    desired_state: DesiredState


class UpdateFlagsRequest(BaseModel):
    """Request to update experiment flags."""

    flags: dict[str, Union[str, int, float, bool]]


class GroupActionRequest(BaseModel):
    """Request to perform an action on all experiments in a group."""

    action: str  # "start", "stop", "terminate"


class ExperimentStatus(BaseModel):
    """Full status of an experiment including current job."""

    experiment: Experiment
    current_job: Optional[Job] = None
    recent_jobs: list[Job] = Field(default_factory=list)


class ExperimentGroup(BaseModel):
    """Group of experiments with shared flag columns.

    Groups allow organizing experiments visually and can define
    which flag columns to display. Experiments can belong to multiple groups.
    """

    id: str = Field(..., description="Unique group ID")
    name: str = Field(..., description="Display name for the group")
    name_prefix: Optional[str] = Field(None, description="Prefix to prepend to experiment names in this group")
    flags: list[str] = Field(default_factory=list, description="Flag columns to display for this group")
    order: int = Field(0, description="Display order of the group")
    created_at: datetime = Field(default_factory=datetime.utcnow, description="When group was created")
    updated_at: datetime = Field(default_factory=datetime.utcnow, description="Last update time")
    collapsed: bool = Field(False, description="Whether group is collapsed in UI")

    # Populated at runtime, not stored
    experiments: list["Experiment"] = Field(default_factory=list, description="Experiments in this group (runtime)")


class ExperimentGroupMembership(BaseModel):
    """Membership of an experiment in a group."""

    group_id: str = Field(..., description="Group ID")
    experiment_id: str = Field(..., description="Experiment ID")
    order: int = Field(0, description="Order within the group")


class CreateGroupRequest(BaseModel):
    """Request to create a new experiment group."""

    name: str
    name_prefix: Optional[str] = None
    flags: list[str] = Field(default_factory=list)


class UpdateGroupRequest(BaseModel):
    """Request to update a group."""

    name: Optional[str] = None
    name_prefix: Optional[str] = None
    flags: Optional[list[str]] = None
    order: Optional[int] = None
    collapsed: Optional[bool] = None


class AddToGroupRequest(BaseModel):
    """Request to add experiments to a group."""

    experiment_ids: list[str]
    multi_home: bool = Field(False, description="If true, add to group without removing from others")


class Checkpoint(BaseModel):
    """Experiment checkpoint with model and replays."""

    experiment_id: Union[int, str] = Field(..., description="Parent experiment ID (integer after migration)")
    epoch: int = Field(..., description="Epoch/step number")
    model_path: Optional[str] = Field(None, description="Path to model file")
    replay_paths: list[str] = Field(default_factory=list, description="List of replay file paths")
    metrics: dict[str, float] = Field(default_factory=dict, description="Training metrics at this epoch")
    created_at: datetime = Field(default_factory=datetime.utcnow, description="When checkpoint was created")
    synced_at: datetime = Field(default_factory=datetime.utcnow, description="When checkpoint was last synced")
    observatory_url: Optional[str] = Field(None, description="Observatory URL for this checkpoint")
    policy_version: Optional[str] = Field(None, description="Policy version number from Observatory")
    policy_id: Optional[str] = Field(None, description="Policy UUID from Observatory")
    policy_version_id: Optional[str] = Field(None, description="Policy version UUID from Observatory")


class BackendStaleness(BaseModel):
    """Staleness information for a backend."""

    last_sync: Optional[datetime] = None
    staleness_seconds: Optional[float] = None
    running: bool = False


class HealthStatus(BaseModel):
    """System health status with per-backend staleness."""

    status: str = "ok"
    skypilot: BackendStaleness = Field(default_factory=BackendStaleness)
    s3: BackendStaleness = Field(default_factory=BackendStaleness)
    observatory: BackendStaleness = Field(default_factory=BackendStaleness)
    num_experiments: int = 0
    num_running_jobs: int = 0
    num_clusters: int = 0


class FlagDefinition(BaseModel):
    """Cached flag definition extracted from a Tool class."""

    tool_path: str = Field(..., description="Tool path (e.g., 'arena.train' or 'metta.tools.train.TrainTool')")
    flag: str = Field(..., description="Flag path (e.g., 'trainer.batch_size')")
    type: str = Field(..., description="Type name (e.g., 'int', 'float', 'str', 'bool')")
    default: Optional[Union[str, int, float, bool]] = Field(None, description="Default value if not required")
    required: bool = Field(False, description="Whether the flag is required")
    last_extracted: datetime = Field(default_factory=datetime.utcnow, description="When flag was last extracted")


class FlagDefinitionsResponse(BaseModel):
    """Response containing flag definitions for a tool path."""

    tool_path: str
    flags: list[FlagDefinition]
