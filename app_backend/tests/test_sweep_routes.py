"""Tests for sweep coordination routes."""

import uuid
from datetime import datetime
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from fastapi.testclient import TestClient

from metta.app_backend.auth import User
from metta.app_backend.metta_repo import MettaRepo, SweepRow
from metta.app_backend.server import create_app


@pytest.fixture
def mock_metta_repo():
    """Create a mock MettaRepo for testing."""
    return MagicMock(spec=MettaRepo)


@pytest.fixture
def test_client(mock_metta_repo: MagicMock):
    """Create a test client with mocked dependencies."""
    app = create_app(mock_metta_repo)
    return TestClient(app)


def test_create_sweep_creates_new(test_client: TestClient, mock_metta_repo: MagicMock, auth_headers: dict[str, str]):
    """Test creating a new sweep."""
    mock_metta_repo.get_sweep_by_name.return_value = None
    test_sweep_id = uuid.uuid4()
    mock_metta_repo.create_sweep.return_value = test_sweep_id

    response = test_client.post(
        "/sweeps/test_sweep/create_sweep",
        json={
            "project": "test_project",
            "entity": "test_entity",
            "wandb_sweep_id": "wandb_123",
        },
        headers=auth_headers,
    )

    assert response.status_code == 200
    assert response.json() == {"created": True, "sweep_id": str(test_sweep_id)}

    # Verify repo methods were called correctly
    mock_metta_repo.get_sweep_by_name.assert_called_once_with("test_sweep")
    mock_metta_repo.create_sweep.assert_called_once_with(
        name="test_sweep",
        project="test_project",
        entity="test_entity",
        wandb_sweep_id="wandb_123",
        user_id="debug_user_id",
    )


def test_create_sweep_returns_existing(
    test_client: TestClient, mock_metta_repo: MagicMock, auth_headers: dict[str, str]
):
    """Test returning existing sweep info (idempotent)."""
    existing_sweep_id = uuid.uuid4()
    existing_sweep = SweepRow(
        id=existing_sweep_id,
        name="test_sweep",
        project="test_project",
        entity="test_entity",
        wandb_sweep_id="wandb_123",
        state="running",
        run_counter=0,
        user_id="test@example.com",
        created_at=datetime.now(),
        updated_at=datetime.now(),
    )
    mock_metta_repo.get_sweep_by_name.return_value = existing_sweep

    response = test_client.post(
        "/sweeps/test_sweep/create_sweep",
        json={
            "project": "test_project",
            "entity": "test_entity",
            "wandb_sweep_id": "wandb_123",
        },
        headers=auth_headers,
    )

    assert response.status_code == 200
    assert response.json() == {"created": False, "sweep_id": str(existing_sweep_id)}

    # Verify only get was called, not create
    mock_metta_repo.get_sweep_by_name.assert_called_once_with("test_sweep")
    mock_metta_repo.create_sweep.assert_not_called()


def test_create_sweep_with_machine_token(test_client: TestClient, mock_metta_repo: MagicMock):
    """Test creating sweep with machine token authentication."""
    mock_metta_repo.get_sweep_by_name.return_value = None
    test_sweep_id = uuid.uuid4()
    mock_metta_repo.create_sweep.return_value = test_sweep_id

    # Mock the login service validation to return a valid user_id
    mock_validate = AsyncMock(return_value=User(id="machine_user_id", email="machine_user@example.com"))

    with patch("metta.app_backend.auth.validate_token_via_login_service", mock_validate):
        response = test_client.post(
            "/sweeps/test_sweep/create_sweep",
            json={
                "project": "test_project",
                "entity": "test_entity",
                "wandb_sweep_id": "wandb_123",
            },
            headers={"X-Auth-Token": "machine_token_123"},
        )

        assert response.status_code == 200
        assert response.json() == {"created": True, "sweep_id": str(test_sweep_id)}

        # Verify the login service was called to validate the token
        mock_validate.assert_called_once_with("machine_token_123")

        # Verify the sweep was created with the user_id from the token validation
        mock_metta_repo.create_sweep.assert_called_once_with(
            name="test_sweep",
            project="test_project",
            entity="test_entity",
            wandb_sweep_id="wandb_123",
            user_id="machine_user_id",
        )


def test_get_sweep_exists(test_client: TestClient, mock_metta_repo: MagicMock, auth_headers: dict[str, str]):
    """Test getting an existing sweep."""
    sweep_id = uuid.uuid4()
    mock_metta_repo.get_sweep_by_name.return_value = SweepRow(
        id=sweep_id,
        name="test_sweep",
        project="test_project",
        entity="test_entity",
        wandb_sweep_id="wandb_123",
        state="running",
        run_counter=0,
        user_id="test@example.com",
        created_at=datetime.now(),
        updated_at=datetime.now(),
    )

    response = test_client.get("/sweeps/test_sweep", headers=auth_headers)

    assert response.status_code == 200
    data = response.json()
    assert data["exists"] is True
    assert data["wandb_sweep_id"] == "wandb_123"


def test_get_sweep_not_exists(test_client: TestClient, mock_metta_repo: MagicMock, auth_headers: dict[str, str]):
    """Test getting a non-existent sweep."""
    mock_metta_repo.get_sweep_by_name.return_value = None

    response = test_client.get("/sweeps/nonexistent", headers=auth_headers)

    assert response.status_code == 200
    data = response.json()
    assert data["exists"] is False
    assert data["wandb_sweep_id"] == ""


def test_get_next_run_id(test_client: TestClient, mock_metta_repo: MagicMock, auth_headers: dict[str, str]):
    """Test getting the next run ID (atomic counter)."""
    sweep_id = uuid.uuid4()
    mock_metta_repo.get_sweep_by_name.return_value = SweepRow(
        id=sweep_id,
        name="test_sweep",
        project="test_project",
        entity="test_entity",
        wandb_sweep_id="wandb_123",
        state="running",
        run_counter=41,  # Will be incremented to 42
        user_id="test@example.com",
        created_at=datetime.now(),
        updated_at=datetime.now(),
    )
    mock_metta_repo.get_next_sweep_run_counter.return_value = 42

    response = test_client.post("/sweeps/test_sweep/runs/next", headers=auth_headers)

    assert response.status_code == 200
    data = response.json()
    assert data["run_id"] == "test_sweep.r.42"

    # Verify the repo was called correctly
    mock_metta_repo.get_sweep_by_name.assert_called_once_with("test_sweep")
    mock_metta_repo.get_next_sweep_run_counter.assert_called_once_with(sweep_id)


def test_get_next_run_id_sweep_not_found(
    test_client: TestClient, mock_metta_repo: MagicMock, auth_headers: dict[str, str]
):
    """Test getting next run ID for non-existent sweep."""
    mock_metta_repo.get_sweep_by_name.return_value = None

    response = test_client.post("/sweeps/nonexistent/runs/next", headers=auth_headers)

    assert response.status_code == 404
    assert "not found" in response.json()["detail"]
