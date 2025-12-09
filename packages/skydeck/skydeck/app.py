"""FastAPI application for SkyDeck dashboard."""

import asyncio
import logging
import os
import subprocess
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Optional

from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles

from .database import Database
from .desired_state import DesiredStateManager
from .flag_schema import get_flag_schemas
from .models import (
    AddToGroupRequest,
    BackendStaleness,
    CreateExperimentRequest,
    CreateGroupRequest,
    DesiredState,
    ExperimentGroup,
    ExperimentStatus,
    HealthStatus,
    UpdateDesiredStateRequest,
    UpdateFlagsRequest,
    UpdateGroupRequest,
)
from .poller import Poller
from .state_manager import StateManager
from .syncer import Syncer

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)

# Global instances
db: Optional[Database] = None
state_manager: Optional[StateManager] = None
desired_state_manager: Optional[DesiredStateManager] = None
poller: Optional[Poller] = None
syncer: Optional[Syncer] = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Lifespan context manager for startup and shutdown."""
    # Startup
    global db, state_manager, desired_state_manager, poller, syncer

    # Get configuration from environment
    from pathlib import Path

    default_db_path = str(Path.home() / ".skydeck" / "skydeck.db")
    db_path = os.getenv("SKYDECK_DB_PATH", default_db_path)

    # Ensure .skydeck directory exists
    Path(db_path).parent.mkdir(parents=True, exist_ok=True)

    poll_interval = int(os.getenv("SKYDECK_POLL_INTERVAL", "30"))
    sync_interval = int(os.getenv("SKYDECK_SYNC_INTERVAL", "60"))

    logger.info("Starting SkyDeck dashboard...")
    logger.info(f"Database: {db_path}")
    logger.info(f"Poll interval: {poll_interval}s")
    logger.info(f"Sync interval: {sync_interval}s")

    # Initialize database
    db = Database(db_path)
    await db.connect()

    # Initialize managers
    state_manager = StateManager(db)
    desired_state_manager = DesiredStateManager(db)

    # Initialize and start background tasks
    poller = Poller(state_manager, interval=poll_interval)
    await poller.start()

    syncer = Syncer(db, interval=sync_interval)
    await syncer.start()

    logger.info("SkyDeck dashboard started successfully")

    yield

    # Shutdown
    logger.info("Shutting down SkyDeck dashboard...")
    if poller:
        await poller.stop()
    if syncer:
        await syncer.stop()
    if db:
        await db.close()
    logger.info("SkyDeck dashboard stopped")


# Create FastAPI app
app = FastAPI(
    title="SkyDeck Dashboard",
    description="Web-based dashboard for managing SkyPilot experiments",
    version="0.1.0",
    lifespan=lifespan,
)

# Mount static files for React build
static_dir = Path(__file__).parent / "static"
assets_dir = static_dir / "assets"
if assets_dir.exists():
    app.mount("/assets", StaticFiles(directory=str(assets_dir)), name="assets")


# Favicon routes
@app.get("/skydeck-favicon.svg")
async def favicon_svg():
    """Serve SVG favicon."""
    favicon_path = static_dir / "skydeck-favicon.svg"
    if favicon_path.exists():
        return FileResponse(favicon_path, media_type="image/svg+xml")
    return {"error": "favicon not found"}


@app.get("/skydeck-favicon.ico")
async def favicon_ico():
    """Serve ICO favicon."""
    favicon_path = static_dir / "skydeck-favicon.ico"
    if favicon_path.exists():
        return FileResponse(favicon_path, media_type="image/x-icon")
    return {"error": "favicon not found"}


# API Endpoints


@app.get("/")
async def index():
    """Serve the main dashboard page."""
    static_file = static_dir / "index.html"
    if static_file.exists():
        return FileResponse(static_file)
    return {"message": "SkyDeck Dashboard - UI not found"}


@app.get("/api/health")
async def health() -> HealthStatus:
    """Get system health status with per-backend staleness."""
    from datetime import datetime

    num_experiments = len(await db.get_all_experiments())
    num_running_jobs = len(await db.get_running_jobs())
    num_clusters = len(await db.get_all_clusters())

    now = datetime.utcnow()

    # Calculate SkyPilot backend staleness
    skypilot_staleness = BackendStaleness(
        running=poller.running if poller else False,
        last_sync=poller.last_poll if poller else None,
    )
    if skypilot_staleness.last_sync:
        skypilot_staleness.staleness_seconds = (now - skypilot_staleness.last_sync).total_seconds()

    # Calculate S3 backend staleness
    s3_staleness = BackendStaleness(
        running=syncer.running if syncer else False,
        last_sync=syncer.last_sync if syncer else None,
    )
    if s3_staleness.last_sync:
        s3_staleness.staleness_seconds = (now - s3_staleness.last_sync).total_seconds()

    # Observatory backend (not yet implemented)
    observatory_staleness = BackendStaleness(running=False, last_sync=None)

    return HealthStatus(
        status="ok",
        skypilot=skypilot_staleness,
        s3=s3_staleness,
        observatory=observatory_staleness,
        num_experiments=num_experiments,
        num_running_jobs=num_running_jobs,
        num_clusters=num_clusters,
    )


# Experiment endpoints


@app.get("/api/experiments")
async def list_experiments():
    """List all experiments."""
    experiments = await desired_state_manager.get_all_experiments()

    # Enrich each experiment with latest epoch from checkpoints
    for exp in experiments:
        exp.latest_epoch = await db.get_latest_epoch(exp.id)

    return {"experiments": experiments}


@app.post("/api/experiments")
async def create_experiment(request: CreateExperimentRequest):
    """Create a new experiment."""
    try:
        experiment = await desired_state_manager.create_experiment(request)
        return {"experiment": experiment}
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e)) from e


@app.get("/api/experiments/{experiment_id}")
async def get_experiment(experiment_id: str):
    """Get experiment by ID."""
    experiment = await desired_state_manager.get_experiment(experiment_id)
    if not experiment:
        raise HTTPException(status_code=404, detail="Experiment not found")
    return {"experiment": experiment}


@app.patch("/api/experiments/{experiment_id}")
async def update_experiment(experiment_id: str, updates: dict):
    """Update experiment configuration fields (name, base_command, git_branch, etc)."""
    experiment = await desired_state_manager.get_experiment(experiment_id)
    if not experiment:
        raise HTTPException(status_code=404, detail="Experiment not found")

    # Handle name update - check uniqueness among non-deleted experiments
    new_name = updates.get("name")
    if new_name and new_name != experiment.name:
        existing = await db.get_experiment_by_name(new_name)
        if existing:
            raise HTTPException(status_code=400, detail="Experiment with that name already exists")
        experiment.name = new_name

    # Update allowed fields
    if "base_command" in updates:
        experiment.base_command = updates["base_command"]
    if "tool_path" in updates:
        experiment.tool_path = updates["tool_path"]
    if "git_branch" in updates:
        experiment.git_branch = updates["git_branch"]
    if "nodes" in updates:
        experiment.nodes = int(updates["nodes"])
    if "gpus" in updates:
        experiment.gpus = int(updates["gpus"])

    # Save updated experiment
    await db.save_experiment(experiment)
    return {"experiment": experiment}


@app.post("/api/experiments/{experiment_id}/expanded")
async def update_experiment_expanded(experiment_id: str, data: dict):
    """Update experiment expanded state."""
    is_expanded = data.get("is_expanded", False)
    await db.update_experiment_expanded(experiment_id, is_expanded)
    return {"status": "ok"}


@app.post("/api/experiments/{experiment_id}/starred")
async def update_experiment_starred(experiment_id: str, data: dict):
    """Update experiment starred state."""
    starred = data.get("starred", False)
    await db._conn.execute("UPDATE experiments SET starred = ? WHERE id = ?", (1 if starred else 0, experiment_id))
    await db._conn.commit()
    return {"status": "ok"}


@app.delete("/api/experiments/{experiment_id}")
async def delete_experiment(experiment_id: str):
    """Soft-delete an experiment."""
    try:
        await desired_state_manager.delete_experiment(experiment_id)
        return {"message": "Experiment deleted"}
    except ValueError as e:
        raise HTTPException(status_code=404, detail=str(e)) from e


@app.post("/api/experiments/{experiment_id}/undelete")
async def undelete_experiment(experiment_id: str):
    """Restore a soft-deleted experiment."""
    try:
        await db.undelete_experiment(experiment_id)
        return {"message": "Experiment restored"}
    except Exception as e:
        raise HTTPException(status_code=404, detail=str(e)) from e


async def launch_experiment(experiment):
    """Launch an experiment by executing its command via subprocess.

    Args:
        experiment: Experiment object to launch
    """
    try:
        # Build the command
        command = experiment.build_command()

        # Replace 'lt' with './devops/skypilot/launch.py' and add --skip-git-check
        if command.startswith("lt "):
            command = "./devops/skypilot/launch.py --skip-git-check " + command[3:]

        logger.info(f"Launching experiment {experiment.name} with command: {command}")

        # Execute the command in a subprocess
        loop = asyncio.get_event_loop()
        await loop.run_in_executor(
            None,
            lambda: subprocess.Popen(
                command,
                shell=True,
                cwd=os.path.expanduser("~/code/metta"),  # Run from metta directory
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
            ),
        )

        logger.info(f"Successfully launched experiment {experiment.name}")
    except Exception as e:
        logger.error(f"Error launching experiment {experiment.name}: {e}", exc_info=True)
        raise


async def stop_experiment(experiment):
    """Stop an experiment by canceling its running job via SkyPilot API.

    Args:
        experiment: Experiment object to stop
    """
    try:
        # Find the current active job for this experiment
        # job.experiment_id is the experiment name
        active_job = await db.get_current_job_for_experiment(experiment.name)

        if active_job and active_job.is_active():
            logger.info(f"Canceling job {active_job.id} for experiment {experiment.name}")

            # Use SkyPilot API to cancel the job
            loop = asyncio.get_event_loop()

            def cancel_job():
                import sky

                sky.cancel(active_job.id)

            await loop.run_in_executor(None, cancel_job)
            logger.info(f"Successfully canceled job {active_job.id} for experiment {experiment.name}")
        else:
            logger.info(f"No active job found for experiment {experiment.name}")
    except Exception as e:
        logger.error(f"Error stopping experiment {experiment.name}: {e}", exc_info=True)
        # Don't raise - we still want to update the desired state even if cancellation fails


@app.post("/api/experiments/{experiment_id}/state")
async def update_desired_state(experiment_id: str, request: UpdateDesiredStateRequest):
    """Update experiment desired state."""
    try:
        # Get current experiment to check if state is already set
        experiment = await desired_state_manager.get_experiment(experiment_id)
        if not experiment:
            raise ValueError(f"Experiment '{experiment_id}' not found")

        # If already in the desired state, do nothing
        if experiment.desired_state == request.desired_state:
            logger.info(f"Experiment {experiment.name} already in desired_state={request.desired_state}, skipping")
            return {"experiment": experiment}

        # Update desired state
        experiment = await desired_state_manager.update_desired_state(experiment_id, request.desired_state)

        # If setting to RUNNING, launch the experiment
        if request.desired_state == DesiredState.RUNNING:
            try:
                await launch_experiment(experiment)
                # Log successful start
                from .models import OperationLog, OperationType
                log = OperationLog(
                    timestamp=datetime.utcnow(),
                    operation_type=OperationType.START,
                    experiment_id=experiment.id,
                    experiment_name=experiment.name,
                    success=True,
                )
                await db.save_operation_log(log)
            except Exception as e:
                # Log failed start
                from .models import OperationLog, OperationType
                log = OperationLog(
                    timestamp=datetime.utcnow(),
                    operation_type=OperationType.START,
                    experiment_id=experiment.id,
                    experiment_name=experiment.name,
                    success=False,
                    error_message=str(e),
                )
                await db.save_operation_log(log)
                raise
        # If setting to STOPPED, cancel the running job
        elif request.desired_state == DesiredState.STOPPED:
            try:
                await stop_experiment(experiment)
                # Log successful stop
                from .models import OperationLog, OperationType
                log = OperationLog(
                    timestamp=datetime.utcnow(),
                    operation_type=OperationType.STOP,
                    experiment_id=experiment.id,
                    experiment_name=experiment.name,
                    success=True,
                )
                await db.save_operation_log(log)
            except Exception as e:
                # Log failed stop
                from .models import OperationLog, OperationType
                log = OperationLog(
                    timestamp=datetime.utcnow(),
                    operation_type=OperationType.STOP,
                    experiment_id=experiment.id,
                    experiment_name=experiment.name,
                    success=False,
                    error_message=str(e),
                )
                await db.save_operation_log(log)
                raise

        return {"experiment": experiment}
    except ValueError as e:
        raise HTTPException(status_code=404, detail=str(e)) from e


@app.get("/api/experiments/{experiment_id}/status")
async def get_experiment_status(experiment_id: str) -> ExperimentStatus:
    """Get full experiment status including current job."""
    experiment = await desired_state_manager.get_experiment(experiment_id)
    if not experiment:
        raise HTTPException(status_code=404, detail="Experiment not found")

    # Get current active job dynamically - use experiment.name since jobs are stored with name as experiment_id
    current_job = await db.get_current_job_for_experiment(experiment.name)

    recent_jobs = await db.get_jobs_for_experiment(experiment.name, limit=10)

    return ExperimentStatus(
        experiment=experiment,
        current_job=current_job,
        recent_jobs=recent_jobs,
    )


@app.get("/api/experiments/{experiment_id}/jobs")
async def get_experiment_jobs(experiment_id: str, limit: int = 10):
    """Get job history for an experiment."""
    experiment = await desired_state_manager.get_experiment(experiment_id)
    if not experiment:
        raise HTTPException(status_code=404, detail="Experiment not found")

    # Use experiment.name since jobs are stored with name as experiment_id
    jobs = await db.get_jobs_for_experiment(experiment.name, limit=limit)
    return {"jobs": jobs}


@app.post("/api/experiments/{experiment_id}/flags")
async def update_experiment_flags(experiment_id: str, request: UpdateFlagsRequest):
    """Update experiment flags."""
    try:
        experiment = await desired_state_manager.update_flags(experiment_id, request.flags)
        return {"experiment": experiment}
    except ValueError as e:
        raise HTTPException(status_code=404, detail=str(e)) from e


@app.get("/api/experiments/{experiment_id}/checkpoints")
async def get_experiment_checkpoints(experiment_id: str, limit: int = 50):
    """Get checkpoints for an experiment."""
    from .services import ObservatoryService

    experiment = await desired_state_manager.get_experiment(experiment_id)
    if not experiment:
        raise HTTPException(status_code=404, detail="Experiment not found")

    # Query checkpoints by experiment.id (integer)
    checkpoints = await db.get_checkpoints(experiment.id, limit=limit)

    # Enrich checkpoints with Observatory URLs if not already set
    for checkpoint in checkpoints:
        if not checkpoint.observatory_url or not checkpoint.policy_version_id:
            if checkpoint.policy_version_id:
                checkpoint.observatory_url = ObservatoryService.get_policy_web_url(checkpoint.policy_version_id)
            else:
                # Policy name in Observatory is the experiment name (not experiment.name.epoch)
                checkpoint.observatory_url = ObservatoryService.get_policy_web_url_by_name(experiment.name)

    return {"checkpoints": checkpoints}


# Job endpoints


@app.post("/api/jobs/{job_id}/cancel")
async def cancel_job(job_id: str):
    """Cancel a running job."""
    job = await db.get_job(job_id)
    if not job:
        raise HTTPException(status_code=404, detail="Job not found")

    if not job.is_active():
        raise HTTPException(status_code=400, detail="Job is not active")

    try:
        # Cancel the job via SkyPilot API
        logger.info(f"Canceling job {job_id}")
        loop = asyncio.get_event_loop()

        def cancel():
            import sky

            sky.cancel(job_id)

        await loop.run_in_executor(None, cancel)
        logger.info(f"Successfully canceled job {job_id}")

        # Also update the experiment's desired state to STOPPED
        # job.experiment_id is the experiment name, not the ID
        experiment = await desired_state_manager.get_experiment_by_name(job.experiment_id)
        if experiment:
            await desired_state_manager.update_desired_state(experiment.id, DesiredState.STOPPED)

        # Log successful cancel
        from .models import OperationLog, OperationType
        log = OperationLog(
            timestamp=datetime.utcnow(),
            operation_type=OperationType.CANCEL,
            experiment_id=experiment.id if experiment else None,
            experiment_name=job.experiment_id,
            job_id=job_id,
            success=True,
        )
        await db.save_operation_log(log)

        return {"message": "Job canceled"}
    except Exception as e:
        logger.error(f"Error canceling job {job_id}: {e}", exc_info=True)

        # Log failed cancel
        from .models import OperationLog, OperationType
        experiment = await desired_state_manager.get_experiment_by_name(job.experiment_id)
        log = OperationLog(
            timestamp=datetime.utcnow(),
            operation_type=OperationType.CANCEL,
            experiment_id=experiment.id if experiment else None,
            experiment_name=job.experiment_id,
            job_id=job_id,
            success=False,
            error_message=str(e),
        )
        await db.save_operation_log(log)

        raise HTTPException(status_code=500, detail=f"Failed to cancel job: {str(e)}") from e


# Operation logs endpoint


@app.get("/api/operation-logs")
async def get_operation_logs(limit: int = 100):
    """Get recent operation logs.

    Args:
        limit: Maximum number of log entries to return (default 100)
    """
    logs = await db.get_operation_logs(limit=limit)
    return {"logs": logs}


# SkyPilot jobs endpoint


@app.get("/api/skypilot-jobs")
async def list_skypilot_jobs(limit: int = 20, include_stopped: bool = False):
    """List all SkyPilot jobs across all clusters.

    Args:
        limit: Maximum number of jobs to return
        include_stopped: If True, include stopped/failed/succeeded jobs
    """
    logger.info(f"list_skypilot_jobs called with limit={limit}, include_stopped={include_stopped}")
    # Get jobs from database with filters
    all_jobs = await db.get_all_jobs(limit=limit, include_stopped=include_stopped)
    logger.info(f"Returning {len(all_jobs)} jobs")

    # Also get clusters to show cluster info
    clusters = await state_manager.get_all_clusters()

    return {"jobs": all_jobs, "clusters": clusters}


# Flag schema endpoints


@app.get("/api/flag-schemas")
async def list_flag_schemas():
    """Get flag schemas for typeahead."""
    schemas = await get_flag_schemas(db)
    return {"schemas": schemas}


# Flag definition endpoints


@app.get("/api/flags")
async def get_flags(tool_path: Optional[str] = None):
    """Get flag definitions extracted from Tool classes via introspection.

    Args:
        tool_path: Tool path (e.g., "arena.train"). If omitted, uses TrainTool defaults.

    Returns:
        Flag definitions with type and default value information
    """
    from .flag_extractor import extract_flags_from_tool_path, get_default_flags

    if not tool_path:
        # Fall back to TrainTool defaults
        flags = get_default_flags()
    else:
        # Extract flags for the specified tool path
        flags = extract_flags_from_tool_path(tool_path)

        # If extraction failed, fall back to TrainTool
        if not flags:
            logger.warning(f"Failed to extract flags from {tool_path}, falling back to TrainTool")
            flags = get_default_flags()

    return {"tool_path": tool_path or "metta.tools.train.TrainTool", "flags": flags}


# System endpoints


@app.post("/api/refresh")
async def force_refresh():
    """Force a refresh of SkyPilot state."""
    if poller:
        await poller.poll_once()
        return {"message": "Refresh complete"}
    raise HTTPException(status_code=503, detail="Poller not running")


@app.post("/api/experiments/reorder")
async def reorder_experiments(request: dict):
    """Update the order of all experiments."""
    experiment_ids = request.get("order", [])

    # Update order for each experiment
    for index, exp_id in enumerate(experiment_ids):
        experiment = await desired_state_manager.get_experiment(exp_id)
        if experiment:
            experiment.order = index
            await db.save_experiment(experiment)

    return {"message": "Order updated"}


# User settings endpoints


@app.get("/api/settings/{key}")
async def get_setting(key: str):
    """Get a user setting by key."""
    value = await db.get_setting(key)
    if value is None:
        raise HTTPException(status_code=404, detail="Setting not found")
    return {"key": key, "value": value}


@app.post("/api/settings/{key}")
async def set_setting(key: str, request: dict):
    """Set a user setting."""
    await db.set_setting(key, request)
    return {"key": key, "value": request}


# Experiment group endpoints


@app.get("/api/groups")
async def list_groups():
    """List all experiment groups with their experiments."""

    groups = await db.get_all_groups()

    # Populate experiments for each group
    for group in groups:
        group.experiments = await db.get_experiments_in_group(group.id)
        # Enrich each experiment with latest epoch
        for exp in group.experiments:
            exp.latest_epoch = await db.get_latest_epoch(exp.id)

    # Get ungrouped experiments
    ungrouped = await db.get_ungrouped_experiments()
    for exp in ungrouped:
        exp.latest_epoch = await db.get_latest_epoch(exp.id)

    return {"groups": groups, "ungrouped": ungrouped}


@app.post("/api/groups")
async def create_group(request: CreateGroupRequest):
    """Create a new experiment group."""
    import uuid
    from datetime import datetime

    # Get max order to place new group at end
    groups = await db.get_all_groups()
    max_order = max((g.order for g in groups), default=-1) + 1

    group = ExperimentGroup(
        id=str(uuid.uuid4())[:8],
        name=request.name,
        name_prefix=request.name_prefix,
        flags=request.flags,
        order=max_order,
        created_at=datetime.utcnow(),
        updated_at=datetime.utcnow(),
    )
    await db.save_group(group)
    return {"group": group}


@app.get("/api/groups/{group_id}")
async def get_group(group_id: str):
    """Get a group by ID with its experiments."""
    group = await db.get_group(group_id)
    if not group:
        raise HTTPException(status_code=404, detail="Group not found")

    group.experiments = await db.get_experiments_in_group(group_id)
    for exp in group.experiments:
        exp.latest_epoch = await db.get_latest_epoch(exp.id)

    return {"group": group}


@app.patch("/api/groups/{group_id}")
async def update_group(group_id: str, request: UpdateGroupRequest):
    """Update a group's properties."""
    from datetime import datetime

    group = await db.get_group(group_id)
    if not group:
        raise HTTPException(status_code=404, detail="Group not found")

    if request.name is not None:
        group.name = request.name
    if request.name_prefix is not None:
        group.name_prefix = request.name_prefix
    if request.flags is not None:
        group.flags = request.flags
    if request.order is not None:
        group.order = request.order
    if request.collapsed is not None:
        group.collapsed = request.collapsed

    group.updated_at = datetime.utcnow()
    await db.save_group(group)
    return {"group": group}


@app.delete("/api/groups/{group_id}")
async def delete_group(group_id: str):
    """Delete a group (experiments are moved to ungrouped)."""
    group = await db.get_group(group_id)
    if not group:
        raise HTTPException(status_code=404, detail="Group not found")

    await db.delete_group(group_id)
    return {"message": "Group deleted"}


@app.post("/api/groups/{group_id}/experiments")
async def add_experiments_to_group(group_id: str, request: AddToGroupRequest):
    """Add experiments to a group."""
    group = await db.get_group(group_id)
    if not group:
        raise HTTPException(status_code=404, detail="Group not found")

    # Get current max order in group
    existing = await db.get_experiments_in_group(group_id)
    max_order = len(existing)

    for i, exp_id in enumerate(request.experiment_ids):
        await db.add_experiment_to_group(group_id, exp_id, order=max_order + i, multi_home=request.multi_home)

    return {"message": f"Added {len(request.experiment_ids)} experiments to group"}


@app.delete("/api/groups/{group_id}/experiments/{experiment_id}")
async def remove_experiment_from_group(group_id: str, experiment_id: str):
    """Remove an experiment from a group."""
    await db.remove_experiment_from_group(group_id, experiment_id)
    return {"message": "Experiment removed from group"}


@app.post("/api/groups/{group_id}/reorder")
async def reorder_experiments_in_group(group_id: str, request: dict):
    """Reorder experiments within a group."""
    experiment_ids = request.get("order", [])
    await db.reorder_experiments_in_group(group_id, experiment_ids)
    return {"message": "Order updated"}


@app.post("/api/groups/reorder")
async def reorder_groups(request: dict):
    """Reorder groups."""
    group_ids = request.get("order", [])
    await db.reorder_groups(group_ids)
    return {"message": "Groups reordered"}
