"""Tests for progressive forgetting curriculum."""

from metta.curriculum.progressive_forgetting import ProgressiveForgettingCurriculum


def test_progressive_forgetting_curriculum_initialization():
    """Test that the curriculum initializes correctly."""
    task_sets = {
        "navigation": {
            "/env/mettagrid/navigation/evals/emptyspace_withinsight": 1,
            "/env/mettagrid/navigation/evals/obstacles1": 1,
        },
        "memory": {
            "/env/mettagrid/memory/evals/easy": 1,
            "/env/mettagrid/memory/evals/medium": 1,
        },
    }

    curriculum = ProgressiveForgettingCurriculum(
        task_sets=task_sets,
        performance_threshold=0.8,
        smoothing=0.1,
        switch_interval=1000,
        eval_interval=100,
        randomize_order=False,
    )

    # Check that all tasks are included
    assert len(curriculum._task_weights) == 4

    # Check that task set mapping is correct
    assert curriculum.task_set_mapping["/env/mettagrid/navigation/evals/emptyspace_withinsight"] == "navigation"
    assert curriculum.task_set_mapping["/env/mettagrid/memory/evals/easy"] == "memory"

    # Check that we start with the first task set
    assert curriculum.current_task_set == "navigation"

    # Check that only navigation tasks have non-zero weights initially
    nav_tasks = [task for task in curriculum._task_weights if "navigation" in task]
    mem_tasks = [task for task in curriculum._task_weights if "memory" in task]

    for task in nav_tasks:
        assert curriculum._task_weights[task] > 0

    for task in mem_tasks:
        assert curriculum._task_weights[task] == 0


def test_progressive_forgetting_curriculum_switching():
    """Test that the curriculum switches task sets correctly."""
    task_sets = {
        "navigation": {
            "/env/mettagrid/navigation/evals/emptyspace_withinsight": 1,
        },
        "memory": {
            "/env/mettagrid/memory/evals/easy": 1,
        },
    }

    curriculum = ProgressiveForgettingCurriculum(
        task_sets=task_sets,
        performance_threshold=0.8,
        smoothing=0.1,
        switch_interval=10,  # Short interval for testing
        eval_interval=5,  # Short interval for testing
        randomize_order=False,
    )

    # Initially should be on navigation
    assert curriculum.current_task_set == "navigation"
    assert curriculum._task_weights["/env/mettagrid/navigation/evals/emptyspace_withinsight"] > 0
    assert curriculum._task_weights["/env/mettagrid/memory/evals/easy"] == 0

    # Complete some tasks with high performance to trigger switch
    for _ in range(20):
        curriculum.complete_task("/env/mettagrid/navigation/evals/emptyspace_withinsight", 0.9)

    # Should have switched to memory
    assert curriculum.current_task_set == "memory"
    assert curriculum._task_weights["/env/mettagrid/navigation/evals/emptyspace_withinsight"] == 0
    assert curriculum._task_weights["/env/mettagrid/memory/evals/easy"] > 0


def test_progressive_forgetting_curriculum_stats():
    """Test that curriculum stats are reported correctly."""
    task_sets = {
        "navigation": {
            "/env/mettagrid/navigation/evals/emptyspace_withinsight": 1,
        },
        "memory": {
            "/env/mettagrid/memory/evals/easy": 1,
        },
    }

    curriculum = ProgressiveForgettingCurriculum(
        task_sets=task_sets,
        performance_threshold=0.8,
        smoothing=0.1,
        switch_interval=1000,
        eval_interval=100,
        randomize_order=False,
    )

    # Complete some tasks
    curriculum.complete_task("/env/mettagrid/navigation/evals/emptyspace_withinsight", 0.5)

    # Get stats
    stats = curriculum.get_curriculum_stats()

    # Check that stats contain expected keys
    assert "current_task_set" in stats
    assert "steps_since_switch" in stats
    assert "steps_since_eval" in stats
    assert "perf_navigation" in stats
    assert "perf_memory" in stats

    # Check that current task set is navigation
    assert stats["current_task_set"] == "navigation"
