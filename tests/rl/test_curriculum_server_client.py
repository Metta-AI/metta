"""Tests for curriculum server/client system."""

import time

from omegaconf import OmegaConf

from metta.mettagrid.curriculum.util import curriculum_from_config_path
from metta.rl.curriculum_client import CurriculumClient
from metta.rl.curriculum_server import CurriculumServer


def test_curriculum_server_client_integration():
    """Test that server and client can communicate properly."""
    
    # Create a simple curriculum
    curriculum = curriculum_from_config_path(
        "/env/mettagrid/arena/8x8", 
        OmegaConf.create({})
    )
    
    # Start the server
    server = CurriculumServer(curriculum, host="localhost", port=15556)
    server.start(background=True)
    
    # Give server time to start
    time.sleep(0.5)
    
    try:
        # Create a client
        client = CurriculumClient(
            server_url="http://localhost:15556",
            batch_size=5
        )
        
        # Test getting a task
        task = client.get_task()
        assert task is not None
        assert hasattr(task, 'env_cfg')
        assert callable(task.env_cfg)
        
        # Test getting env configs
        configs = client.get_env_cfg_by_bucket()
        assert isinstance(configs, dict)
        assert len(configs) > 0
        
        # Test stats methods return empty
        assert client.get_completion_rates() == {}
        assert client.get_task_probs() == {}
        assert client.get_curriculum_stats() == {}
        
        # Test complete_task is no-op
        client.complete_task("test", 0.5)  # Should not raise
        
        # Test multiple tasks are randomly selected
        task_ids = []
        for _ in range(20):
            task = client.get_task()
            task_id = task.id if hasattr(task, 'id') else task.name
            task_ids.append(task_id)
        
        # With batch size 5 and 20 selections, we should see some repeats
        # if random selection is working
        assert len(set(task_ids)) <= 5
        
    finally:
        # Shutdown server
        server.stop()


def test_curriculum_client_conforms_to_interface():
    """Test that CurriculumClient properly implements Curriculum interface."""
    from metta.mettagrid.curriculum.core import Curriculum
    
    # Check that CurriculumClient is a subclass of Curriculum
    assert issubclass(CurriculumClient, Curriculum)
    
    # Check that all required methods are present
    required_methods = [
        'get_task',
        'get_env_cfg_by_bucket',
        'complete_task',
        'get_completion_rates',
        'get_task_probs',
        'get_curriculum_stats'
    ]
    
    for method in required_methods:
        assert hasattr(CurriculumClient, method)
        assert callable(getattr(CurriculumClient, method))