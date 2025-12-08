"""FastAPI application for SkyDeck dashboard."""

import logging
import os
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
    BackendStaleness,
    CreateExperimentRequest,
    ExperimentStatus,
    HealthStatus,
    UpdateDesiredStateRequest,
    UpdateFlagsRequest,
)
from .poller import Poller
from .reconciler import Reconciler
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
reconciler: Optional[Reconciler] = None
syncer: Optional[Syncer] = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Lifespan context manager for startup and shutdown."""
    # Startup
    global db, state_manager, desired_state_manager, poller, reconciler, syncer

    # Get configuration from environment
    from pathlib import Path

    default_db_path = str(Path.home() / ".skydeck" / "skydeck.db")
    db_path = os.getenv("SKYDECK_DB_PATH", default_db_path)

    # Ensure .skydeck directory exists
    Path(db_path).parent.mkdir(parents=True, exist_ok=True)

    poll_interval = int(os.getenv("SKYDECK_POLL_INTERVAL", "30"))
    reconcile_interval = int(os.getenv("SKYDECK_RECONCILE_INTERVAL", "60"))
    sync_interval = int(os.getenv("SKYDECK_SYNC_INTERVAL", "60"))

    logger.info("Starting SkyDeck dashboard...")
    logger.info(f"Database: {db_path}")
    logger.info(f"Poll interval: {poll_interval}s")
    logger.info(f"Reconcile interval: {reconcile_interval}s")
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

    reconciler = Reconciler(db, desired_state_manager, state_manager, interval=reconcile_interval)
    await reconciler.start()

    syncer = Syncer(db, interval=sync_interval)
    await syncer.start()

    logger.info("SkyDeck dashboard started successfully")

    yield

    # Shutdown
    logger.info("Shutting down SkyDeck dashboard...")
    if poller:
        await poller.stop()
    if reconciler:
        await reconciler.stop()
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

# Mount static files
static_dir = Path(__file__).parent / "static"
if static_dir.exists():
    app.mount("/static", StaticFiles(directory=str(static_dir)), name="static")


# API Endpoints


@app.get("/")
async def index():
    """Serve the main dashboard page."""
    static_file = Path(__file__).parent / "static" / "index.html"
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
    """Update experiment configuration fields (base_command, git_branch, id)."""
    experiment = await desired_state_manager.get_experiment(experiment_id)
    if not experiment:
        raise HTTPException(status_code=404, detail="Experiment not found")

    # Handle ID rename separately
    new_id = updates.get("id")
    if new_id and new_id != experiment_id:
        # Check if new ID already exists
        existing = await desired_state_manager.get_experiment(new_id)
        if existing:
            raise HTTPException(status_code=400, detail="Experiment with that ID already exists")

        # Rename experiment
        await db.rename_experiment(experiment_id, new_id)
        experiment_id = new_id
        experiment = await desired_state_manager.get_experiment(new_id)

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


@app.post("/api/experiments/{experiment_id}/state")
async def update_desired_state(experiment_id: str, request: UpdateDesiredStateRequest):
    """Update experiment desired state."""
    try:
        experiment = await desired_state_manager.update_desired_state(experiment_id, request.desired_state)
        return {"experiment": experiment}
    except ValueError as e:
        raise HTTPException(status_code=404, detail=str(e)) from e


@app.get("/api/experiments/{experiment_id}/status")
async def get_experiment_status(experiment_id: str) -> ExperimentStatus:
    """Get full experiment status including current job."""
    experiment = await desired_state_manager.get_experiment(experiment_id)
    if not experiment:
        raise HTTPException(status_code=404, detail="Experiment not found")

    # Get current active job dynamically
    current_job = await db.get_current_job_for_experiment(experiment_id)

    recent_jobs = await db.get_jobs_for_experiment(experiment_id, limit=10)

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

    jobs = await db.get_jobs_for_experiment(experiment_id, limit=limit)
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

    checkpoints = await db.get_checkpoints(experiment_id, limit=limit)

    # Enrich checkpoints with Observatory URLs
    for checkpoint in checkpoints:
        policy_name = f"{experiment_id}.{checkpoint.epoch}"
        checkpoint.observatory_url = ObservatoryService.get_policy_web_url(policy_name)

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

    # Let the reconciler handle cancellation by setting desired state to STOPPED
    experiment = await desired_state_manager.get_experiment(job.experiment_id)
    if experiment:
        from .models import DesiredState

        await desired_state_manager.update_desired_state(job.experiment_id, DesiredState.STOPPED)

    return {"message": "Job cancellation requested"}


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


# System endpoints


@app.post("/api/refresh")
async def force_refresh():
    """Force a refresh of SkyPilot state."""
    if poller:
        await poller.poll_once()
        return {"message": "Refresh complete"}
    raise HTTPException(status_code=503, detail="Poller not running")


@app.post("/api/reconcile")
async def force_reconcile():
    """Force a reconciliation cycle."""
    if reconciler:
        await reconciler.reconcile_once()
        return {"message": "Reconciliation complete"}
    raise HTTPException(status_code=503, detail="Reconciler not running")


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
