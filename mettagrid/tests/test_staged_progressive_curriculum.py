import pytest
from omegaconf import OmegaConf

from mettagrid.curriculum import StagedProgressiveCurriculum


@pytest.fixture
def simple_env_cfg():
    return OmegaConf.create({
        "sampling": 0, 
        "game": {"num_agents": 1, "map": {"width": 10, "height": 10}}
    })


def test_staged_progressive_curriculum_initialization():
    """Test that StagedProgressiveCurriculum initializes correctly."""
    stages = [
        {"curriculum": "test_stage_1", "name": "stage1", "weight": 1.0},
        {"curriculum": "test_stage_2", "name": "stage2", "weight": 1.0},
    ]
    
    # Mock the curriculum_from_config_path to return a simple curriculum
    def mock_curriculum_from_config_path(path, env_overrides):
        from mettagrid.curriculum import SingleTaskCurriculum
        return SingleTaskCurriculum(path, simple_env_cfg())
    
    # Patch the import
    import mettagrid.curriculum.staged_progressive as staged_prog
    original_func = staged_prog.curriculum_from_config_path
    staged_prog.curriculum_from_config_path = mock_curriculum_from_config_path
    
    try:
        curriculum = StagedProgressiveCurriculum(
            stages=stages,
            transition_criteria="performance",
            performance_threshold=0.8
        )
        
        assert curriculum._current_stage == 0
        assert len(curriculum._stage_curricula) == 2
        assert curriculum._stage_names == ["stage1", "stage2"]
        
    finally:
        # Restore original function
        staged_prog.curriculum_from_config_path = original_func


def test_staged_progressive_curriculum_transition():
    """Test that stage transitions work correctly."""
    stages = [
        {"curriculum": "test_stage_1", "name": "stage1", "weight": 1.0},
        {"curriculum": "test_stage_2", "name": "stage2", "weight": 1.0},
    ]
    
    def mock_curriculum_from_config_path(path, env_overrides):
        from mettagrid.curriculum import SingleTaskCurriculum
        return SingleTaskCurriculum(path, simple_env_cfg())
    
    import mettagrid.curriculum.staged_progressive as staged_prog
    original_func = staged_prog.curriculum_from_config_path
    staged_prog.curriculum_from_config_path = mock_curriculum_from_config_path
    
    try:
        curriculum = StagedProgressiveCurriculum(
            stages=stages,
            transition_criteria="performance",
            performance_threshold=0.8
        )
        
        # Should start at stage 0
        assert curriculum._current_stage == 0
        
        # Complete tasks with high performance to trigger transition
        for _ in range(20):  # Enough to reach threshold with smoothing
            curriculum.complete_task("test_task", 0.9)
        
        # Should have transitioned to stage 1
        assert curriculum._current_stage == 1
        
    finally:
        staged_prog.curriculum_from_config_path = original_func


def test_staged_progressive_curriculum_time_based_transition():
    """Test time-based stage transitions."""
    stages = [
        {"curriculum": "test_stage_1", "name": "stage1", "weight": 1.0},
        {"curriculum": "test_stage_2", "name": "stage2", "weight": 1.0},
    ]
    
    def mock_curriculum_from_config_path(path, env_overrides):
        from mettagrid.curriculum import SingleTaskCurriculum
        return SingleTaskCurriculum(path, simple_env_cfg())
    
    import mettagrid.curriculum.staged_progressive as staged_prog
    original_func = staged_prog.curriculum_from_config_path
    staged_prog.curriculum_from_config_path = mock_curriculum_from_config_path
    
    try:
        curriculum = StagedProgressiveCurriculum(
            stages=stages,
            transition_criteria="time",
            time_threshold_steps=1000
        )
        
        # Should start at stage 0
        assert curriculum._current_stage == 0
        
        # Update step count to trigger transition
        curriculum.update_step_count(1000)
        
        # Complete a task to check for transition
        curriculum.complete_task("test_task", 0.5)
        
        # Should have transitioned to stage 1
        assert curriculum._current_stage == 1
        
    finally:
        staged_prog.curriculum_from_config_path = original_func 