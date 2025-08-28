from dataclasses import dataclass
import logging

from cogweb.cogweb_client import CogwebClient
from pydantic import ConfigDict
from enum import StrEnum, auto

from metta.common.wandb.wandb_context import WandbConfig
from typing import Callable, Any

logger = logging.getLogger(__name__)


# Job definition
@dataclass
class JobDefinition:
    run: str
    type: str = "train"


class DispatchType(StrEnum):
    LOCAL = auto()
    SKYPILOT = auto()
    CENTRAL_QUEUE = auto()


# Base Classes
class Scheduler:
    def __init__(self):
        pass

    def schedule(self) -> list[JobDefinition]:
        new_job = JobDefinition(run="trainin_run")
        return [new_job]


class Store:
    def __init__(self):
        pass

    def fetch_active_jobs(self) -> list[str]:
        return ["run_1"]

    def fetch_pending_jobs(self) -> list[str]:
        return ["run_0"]


class Dispatcher:
    def __init__(self):
        pass

    def dispatch_job(self, job: JobDefinition, dispatch_type: DispatchType) -> bool:
        if dispatch_type is DispatchType.LOCAL:
            # TODO: Dispatch Local Job
            logger.info(f"Dispatching Job: {job.run} ({job.type}). Dispatch type: {dispatch_type.value}")
            return True
        elif dispatch_type is DispatchType.SKYPILOT:
            # TODO: Dispatch to skypilot
            return True
        elif dispatch_type is DispatchType.CENTRAL_QUEUE:
            # TODO: Dispatch to central queue
            return True


class Optimizer:
    def __init__(self):
        self.observations = []

    def tell(self):
        pass

    def ask(self):
        pass


class SweepOrchestratorConfig(ConfigDict):
    sweep_name: str
    sweep_server_uri: str
    wandb: WandbConfig


class SweepState:
    def __init__(self):
        num_jobs = 0
        num_active_jobs = 0
        num_pending_jobs = 0
        num_complete_jobs = 0

        failed_jobs = list[JobDefinition]


def start_watch(func: Callable, until: Callable[[Any], bool], with_delay: float):
    while until is False:
        pass
        # Execute function
        # waith for with_delay seconds


def orchestrate_sweep(
    sweep_name: str,
    sweep_server_uri: str,
    wandb: WandbConfig,
    scheduler: Scheduler,
    optimizer: Optimizer,
    dispatcher: Dispatcher,
):
    # Step 1: Initialize sweep services
    cogweb_client = CogwebClient.get_client(base_url=sweep_server_uri)
    sweep_client = cogweb_client.sweep_client()

    # Register sweep if it doesn't exist
    sweep_info = sweep_client.get_sweep(sweep_name)
    if not sweep_info.exists:
        logger.info(f"Registering sweep {sweep_name}")
        sweep_client.create_sweep(sweep_name, wandb.project, wandb.entity, sweep_name)

    # Step 2
    watch()


def inner_sweep_loop(sweep_state: SweepState, scheduler: Scheduler, dispatcher: Dispatcher):
    jobs = Scheduler()
    for job in jobs:
        dispatcher.dispatch_job(job, dispatch_type=DispatchType.SKYPILOT)
