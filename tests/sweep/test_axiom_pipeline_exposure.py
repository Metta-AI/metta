"""Unit tests for Pipeline exposure and override features."""

import pytest
from typing import Any

from metta.sweep.axiom.core import Pipeline, Ctx


class TestPipelineExposure:
    """Test the Pipeline exposure and override features."""

    def test_stage_not_exposed_by_default(self):
        """Test that stages are not exposed by default."""
        pipeline = Pipeline()
        pipeline.stage("test", lambda x: x * 2)
        
        assert "test" not in pipeline._exposed
        assert len(pipeline._exposed) == 0

    def test_stage_exposed_when_specified(self):
        """Test that stages can be explicitly exposed."""
        pipeline = Pipeline()
        pipeline.stage("test", lambda x: x * 2, expose=True)
        
        assert "test" in pipeline._exposed
        assert len(pipeline._exposed) == 1

    def test_io_not_exposed_by_default(self):
        """Test that I/O operations are not exposed by default."""
        pipeline = Pipeline()
        pipeline.io("save", lambda x: x)
        
        assert "save" not in pipeline._exposed
        assert len(pipeline._exposed) == 0

    def test_io_exposed_when_specified(self):
        """Test that I/O operations can be explicitly exposed."""
        pipeline = Pipeline()
        pipeline.io("save", lambda x: x, expose=True)
        
        assert "save" in pipeline._exposed
        assert len(pipeline._exposed) == 1

    def test_join_exposed_by_default(self):
        """Test that joins are exposed by default."""
        sub_pipeline = Pipeline()
        sub_pipeline.stage("sub_stage", lambda x: x + 1)
        
        pipeline = Pipeline()
        pipeline.join("sub", sub_pipeline)
        
        assert "sub" in pipeline._exposed

    def test_join_not_exposed_when_disabled(self):
        """Test that joins can be set to not exposed."""
        sub_pipeline = Pipeline()
        sub_pipeline.stage("sub_stage", lambda x: x + 1)
        
        pipeline = Pipeline()
        pipeline.join("sub", sub_pipeline, expose=False)
        
        assert "sub" not in pipeline._exposed

    def test_override_exposed_component(self):
        """Test overriding an exposed component."""
        pipeline = Pipeline()
        pipeline.stage("compute", lambda x: x * 2, expose=True)
        
        # Override with a different function
        pipeline.override("compute", lambda x: x * 3)
        
        assert "compute" in pipeline._overrides
        assert pipeline._overrides["compute"](5) == 15

    def test_override_non_exposed_component_fails(self):
        """Test that overriding non-exposed component raises error."""
        pipeline = Pipeline()
        pipeline.stage("internal", lambda x: x * 2)  # Not exposed
        
        with pytest.raises(ValueError, match="not exposed"):
            pipeline.override("internal", lambda x: x * 3)

    def test_override_nested_component(self):
        """Test overriding a nested component through join path."""
        # Create sub-pipeline with exposed stage
        sub_pipeline = Pipeline()
        sub_pipeline.stage("compute", lambda x: x * 2, expose=True)
        
        # Create main pipeline with exposed join
        pipeline = Pipeline()
        pipeline.join("sub", sub_pipeline, expose=True)
        
        # Override nested component
        pipeline.override("sub.compute", lambda x: x * 10)
        
        # Check that override propagated
        assert "compute" in sub_pipeline._overrides

    def test_override_through_non_exposed_join_fails(self):
        """Test that override through non-exposed join fails."""
        sub_pipeline = Pipeline()
        sub_pipeline.stage("compute", lambda x: x * 2, expose=True)
        
        pipeline = Pipeline()
        pipeline.join("sub", sub_pipeline, expose=False)  # Not exposed
        
        with pytest.raises(ValueError, match="join not exposed"):
            pipeline.override("sub.compute", lambda x: x * 10)

    def test_list_exposed_components(self):
        """Test listing all exposed components."""
        sub_pipeline = Pipeline()
        sub_pipeline.stage("sub_compute", lambda x: x, expose=True)
        sub_pipeline.stage("sub_internal", lambda x: x)  # Not exposed
        
        pipeline = Pipeline()
        pipeline.stage("main_compute", lambda x: x, expose=True)
        pipeline.stage("main_internal", lambda x: x)  # Not exposed
        pipeline.join("sub", sub_pipeline, expose=True)
        
        exposed = pipeline.list_exposed()
        
        assert "main_compute" in exposed
        assert "sub" in exposed
        assert "sub.sub_compute" in exposed
        assert "main_internal" not in exposed
        assert "sub.sub_internal" not in exposed

    def test_get_overrides(self):
        """Test getting all active overrides."""
        sub_pipeline = Pipeline()
        sub_pipeline.stage("compute", lambda x: x * 2, expose=True)
        
        pipeline = Pipeline()
        pipeline.stage("process", lambda x: x + 1, expose=True)
        pipeline.join("sub", sub_pipeline, expose=True)
        
        # Add overrides
        new_process = lambda x: x + 10
        new_compute = lambda x: x * 20
        
        pipeline.override("process", new_process)
        pipeline.override("sub.compute", new_compute)
        
        overrides = pipeline.get_overrides()
        
        assert overrides["process"] is new_process
        assert overrides["sub.compute"] is new_compute

    def test_callable_wrapped_as_pipeline_in_join(self):
        """Test that callable is wrapped as Pipeline in join."""
        pipeline = Pipeline()
        func = lambda x: x * 2
        pipeline.join("compute", func)
        
        # Check that it was wrapped
        assert "compute" in pipeline._provided_joins
        assert isinstance(pipeline._provided_joins["compute"], Pipeline)


class TestPipelineExecutionWithOverrides:
    """Test that overrides work correctly during execution."""

    def test_simple_override_execution(self):
        """Test that override is used during execution."""
        pipeline = Pipeline()
        pipeline.stage("init", lambda _: 10)
        pipeline.stage("double", lambda x: x * 2, expose=True)
        
        # Run without override
        ctx1 = Ctx()
        result1 = pipeline.run(ctx1)
        assert result1 == 20
        
        # Override and run again
        pipeline.override("double", lambda x: x * 3)
        ctx2 = Ctx()
        result2 = pipeline.run(ctx2)
        assert result2 == 30

    def test_nested_override_execution(self):
        """Test that nested override works during execution."""
        # Create sub-pipeline
        sub = Pipeline()
        sub.stage("add", lambda x: x + 1, expose=True)
        
        # Create main pipeline
        main = Pipeline()
        main.stage("init", lambda _: 5)
        main.join("sub", sub, expose=True)
        main.stage("final", lambda x: x * 2)
        
        # Run without override
        result1 = main.run()
        assert result1 == (5 + 1) * 2  # 12
        
        # Override nested component
        main.override("sub.add", lambda x: x + 10)
        result2 = main.run()
        assert result2 == (5 + 10) * 2  # 30

    def test_multiple_overrides(self):
        """Test multiple overrides work correctly."""
        pipeline = Pipeline()
        pipeline.stage("a", lambda _: 1, expose=True)
        pipeline.stage("b", lambda x: x + 2, expose=True)
        pipeline.stage("c", lambda x: x * 3, expose=True)
        
        # Original execution
        result1 = pipeline.run()
        assert result1 == (1 + 2) * 3  # 9
        
        # Override multiple stages
        pipeline.override("a", lambda _: 10)
        pipeline.override("c", lambda x: x * 10)
        
        result2 = pipeline.run()
        assert result2 == (10 + 2) * 10  # 120