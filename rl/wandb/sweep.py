import wandb
import os
import logging

logger = logging.getLogger("sweep")

def sweep_id_from_name(project: str, name: str) -> str:
    api = wandb.Api()
    project = api.project(project)
    sweeps = project.sweeps()
    for sweep in sweeps:
        if sweep.name == name:
            return sweep.id
    return None

def generate_run_id_for_sweep(sweep_id: str, sweep_dir: str) -> str:
    api = wandb.Api()
    sweep = api.sweep(sweep_id)

    used_ids = set()
    used_names = set(run.name for run in sweep.runs).union(
        set(os.listdir(os.path.join(sweep_dir, "runs"))))
    for name in used_names:
        try:
            id = int(name.split('.')[-1])
            used_ids.add(id)
        except ValueError:
            logger.warning(f"Invalid run name: {name}, not ending with an integer")

    id = 0
    if len(used_ids) > 0:
        id = max(used_ids) + 1

    return f"{sweep.name}.r.{id}"
