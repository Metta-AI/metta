from metta.app_backend.routes.sweep_routes import SweepCreateResponse
from metta.app_backend.sweep_client import SweepClient


def get_sweep_id_from_metta(sweep_name: str) -> str | None:
    client = SweepClient()
    info = client.get_sweep(sweep_name)
    if info.exists:
        return info.wandb_sweep_id
    else:
        return None


def get_next_run_id_from_metta(sweep_name: str) -> str:
    client = SweepClient()
    return client.get_next_run_id(sweep_name)


def create_sweep_in_metta(sweep_name: str, entity: str, project: str, wandb_sweep_id: str) -> SweepCreateResponse:
    client = SweepClient()
    return client.create_sweep(sweep_name, entity, project, wandb_sweep_id)
