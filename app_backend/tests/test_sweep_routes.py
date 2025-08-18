"""Tests for sweep coordination routes."""

import uuid
from datetime import datetime
from unittest.mock import MagicMock

import pytest
from fastapi.testclient import TestClient

from metta.app_backend.metta_repo import MettaRepo, SweepRow
from metta.app_backend.server import create_app


@pytest.fixture
def mock_metta_repo():
    """Create a mock MettaRepo for testing."""
    return MagicMock(spec=MettaRepo)


@pytest.fixture
def test_client(mock_metta_repo):
    """Create a test client with mocked dependencies."""
    app = create_app(mock_metta_repo)
    return TestClient(app)


def test_create_sweep_creates_new(test_client, mock_metta_repo, auth_headers):
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
        user_id="test@example.com",
    )


def test_create_sweep_returns_existing(test_client, mock_metta_repo, auth_headers):
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


def test_create_sweep_with_machine_token(test_client, mock_metta_repo):
    """Test creating sweep with machine token authentication."""
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
        headers={"X-Auth-Token": "machine_token_123"},
    )

    assert response.status_code == 200
    assert response.json() == {"created": True, "sweep_id": str(test_sweep_id)}


def test_get_sweep_exists(test_client, mock_metta_repo, auth_headers):
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


def test_get_sweep_not_exists(test_client, mock_metta_repo, auth_headers):
    """Test getting a non-existent sweep."""
    mock_metta_repo.get_sweep_by_name.return_value = None

    response = test_client.get("/sweeps/nonexistent", headers=auth_headers)

    assert response.status_code == 200
    data = response.json()
    assert data["exists"] is False
    assert data["wandb_sweep_id"] == ""


def test_get_next_run_id(test_client, mock_metta_repo, auth_headers):
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


def test_get_next_run_id_sweep_not_found(test_client, mock_metta_repo, auth_headers):
    """Test getting next run ID for non-existent sweep."""
    mock_metta_repo.get_sweep_by_name.return_value = None

    response = test_client.post("/sweeps/nonexistent/runs/next", headers=auth_headers)

    assert response.status_code == 404
    assert "not found" in response.json()["detail"]


def test_auth_required_for_all_operations(test_client, mock_metta_repo):
    """Test that all operations require authentication."""
    # Test create_sweep without auth
    response = test_client.post(
        "/sweeps/test_sweep/create_sweep",
        json={
            "project": "test_project",
            "entity": "test_entity",
            "wandb_sweep_id": "wandb_123",
        },
    )
    assert response.status_code == 401

    # Test get_sweep without auth
    response = test_client.get("/sweeps/test_sweep")
    assert response.status_code == 401

    # Test get_next_run_id without auth
    response = test_client.post("/sweeps/test_sweep/runs/next")
    assert response.status_code == 401
