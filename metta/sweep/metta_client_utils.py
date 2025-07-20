from metta.app_backend.routes.sweep_routes import SweepCreateResponse
from metta.app_backend.sweep_client import SweepClient
from metta.common.util.stats_client_cfg import get_machine_token


def _get_sweep_client(backend_url: str | None = None) -> SweepClient:
    """Create a SweepClient with proper authentication and URL configuration.

    Args:
        backend_url: Backend URL. If None, defaults to localhost for backward compatibility.

    Returns:
        Configured SweepClient instance
    """
    # Default to localhost for backward compatibility
    if backend_url is None:
        backend_url = "http://localhost:8000"

    # Get machine token for authentication (same pattern as stats client)
    auth_token = get_machine_token(backend_url)

    return SweepClient(base_url=backend_url, auth_token=auth_token)


def get_sweep_id_from_metta(sweep_name: str, backend_url: str | None = None) -> str | None:
    """Get sweep ID from centralized database.

    Args:
        sweep_name: Name of the sweep
        backend_url: Backend URL override (optional, defaults to localhost)
    """
    client = _get_sweep_client(backend_url)
    info = client.get_sweep(sweep_name)
    if info.exists:
        return info.wandb_sweep_id
    else:
        return None


def get_next_run_id_from_metta(sweep_name: str, backend_url: str | None = None) -> str:
    """Get next run ID from centralized database (atomic operation).

    Args:
        sweep_name: Name of the sweep
        backend_url: Backend URL override (optional, defaults to localhost)
    """
    client = _get_sweep_client(backend_url)
    return client.get_next_run_id(sweep_name)


def create_sweep_in_metta(
    sweep_name: str,
    entity: str,
    project: str,
    wandb_sweep_id: str,
    backend_url: str | None = None,
) -> SweepCreateResponse:
    """Create sweep in centralized database (idempotent).

    Args:
        sweep_name: Name of the sweep
        entity: WandB entity
        project: WandB project
        wandb_sweep_id: WandB sweep ID
        backend_url: Backend URL override (optional, defaults to localhost)
    """
    client = _get_sweep_client(backend_url)
    return client.create_sweep(sweep_name, project, entity, wandb_sweep_id)
