"""Unit tests for tAXIOM core components - MVP version."""

import pytest
from typing import Any

from metta.sweep.axiom.core import Ctx, Pipeline, Stage


class TestContext:
    """Test the Context (Ctx) class."""

    def test_ctx_initialization(self):
        """Test that context initializes with empty collections."""
        ctx = Ctx()
        assert ctx.stages == {}
        assert ctx.metadata == {}
        assert ctx.artifacts == {}
        assert ctx._current_stage is None

    def test_stage_input_output(self):
        """Test setting and getting stage inputs/outputs."""
        ctx = Ctx()
        
        # Set input for a stage
        ctx.set_stage_input("test_stage", {"data": "input"})
        assert ctx.get_stage_input("test_stage") == {"data": "input"}
        
        # Set output for a stage
        ctx.set_stage_output("test_stage", {"result": "output"})
        assert ctx.get_stage_output("test_stage") == {"result": "output"}
        
        # Check that both are stored
        assert "in" in ctx.stages["test_stage"]
        assert "out" in ctx.stages["test_stage"]

    def test_get_last_output(self):
        """Test getting the last stage's output."""
        ctx = Ctx()
        
        # Initially no last output
        assert ctx.get_last_output() is None
        
        # Set output and current stage
        ctx.set_stage_output("stage1", "result1")
        ctx._current_stage = "stage1"
        assert ctx.get_last_output() == "result1"
        
        # Update to new stage
        ctx.set_stage_output("stage2", "result2")
        ctx._current_stage = "stage2"
        assert ctx.get_last_output() == "result2"

    def test_metadata_usage(self):
        """Test that metadata can be used for arbitrary data."""
        ctx = Ctx()
        ctx.metadata["seed"] = 42
        ctx.metadata["trial_id"] = 1
        
        assert ctx.metadata["seed"] == 42
        assert ctx.metadata["trial_id"] == 1


class TestStage:
    """Test the Stage class."""

    def test_stage_initialization(self):
        """Test basic stage creation."""
        def dummy_func(x):
            return x * 2
        
        stage = Stage(name="test", func=dummy_func)
        assert stage.name == "test"
        assert stage.func == dummy_func
        assert stage.stage_type == "stage"
        assert stage.input_type is None
        assert stage.output_type is None
        assert stage.hooks == []

    def test_stage_with_types(self):
        """Test stage with type annotations."""
        def typed_func(x: int) -> str:
            return str(x)
        
        stage = Stage(
            name="typed",
            func=typed_func,
            input_type=int,
            output_type=str
        )
        assert stage.input_type == int
        assert stage.output_type == str

    def test_stage_execute_simple(self):
        """Test executing a simple stage."""
        def double(x):
            return x * 2
        
        stage = Stage(name="double", func=double)
        ctx = Ctx()
        
        # Execute with explicit args
        result = stage.execute(ctx, 5)
        assert result == 10
        assert ctx.get_stage_output("double") == 10

    def test_stage_execute_from_context(self):
        """Test executing a stage that reads input from context."""
        def triple(x):
            return x * 3
        
        stage = Stage(name="triple", func=triple)
        ctx = Ctx()
        ctx.set_stage_input("triple", 4)
        
        # Execute without args - should read from context
        result = stage.execute(ctx)
        assert result == 12
        assert ctx.get_stage_output("triple") == 12

    def test_config_access_patterns(self):
        """Test different patterns for accessing configuration."""
        # Pattern 1: Method with self.cfg
        class Experiment:
            def __init__(self, multiplier):
                self.cfg = {"multiplier": multiplier}
            
            def process(self, data):
                return data * self.cfg["multiplier"]
        
        exp = Experiment(multiplier=3)
        stage = Stage(name="method", func=exp.process)
        ctx = Ctx()
        result = stage.execute(ctx, 5)
        assert result == 15
        
        # Pattern 2: Lambda with closed-over config
        config_value = 2
        stage = Stage(name="lambda", func=lambda x: x * config_value)
        result = stage.execute(ctx, 10)
        assert result == 20
        
        # Pattern 3: Partial application
        from functools import partial
        
        def multiply(data, factor):
            return data * factor
        
        stage = Stage(name="partial", func=partial(multiply, factor=4))
        result = stage.execute(ctx, 3)
        assert result == 12

    def test_stage_with_hooks(self):
        """Test that hooks are stored but not executed by stage."""
        hook_calls = []
        
        def hook(result, ctx):
            hook_calls.append(result)
        
        stage = Stage(name="test", func=lambda x: x, hooks=[hook])
        assert len(stage.hooks) == 1
        
        # Stage.execute doesn't run hooks - Pipeline does
        ctx = Ctx()
        stage.execute(ctx, "data")
        assert hook_calls == []  # Hook not called by stage itself

    def test_stage_type_io(self):
        """Test creating an I/O stage."""
        def fetch_data():
            return {"fetched": "data"}
        
        stage = Stage(name="fetch", func=fetch_data, stage_type="io")
        assert stage.stage_type == "io"

    def test_invalid_stage_func(self):
        """Test that non-callable func raises error."""
        with pytest.raises(TypeError, match="func must be callable"):
            Stage(name="bad", func="not a function")


class TestPipeline:
    """Test the Pipeline class."""

    def test_pipeline_initialization(self):
        """Test pipeline starts empty."""
        pipeline = Pipeline()
        assert pipeline.stages == []
        assert pipeline.stage_names == set()
        assert pipeline._built is False
        assert pipeline._ctx is None

    def test_add_stage(self):
        """Test adding stages to pipeline."""
        pipeline = Pipeline()
        
        pipeline.stage("first", lambda x: x + 1)
        assert len(pipeline.stages) == 1
        assert "first" in pipeline.stage_names
        
        pipeline.stage("second", lambda x: x * 2)
        assert len(pipeline.stages) == 2
        assert "second" in pipeline.stage_names

    def test_add_io(self):
        """Test adding I/O operations to pipeline."""
        pipeline = Pipeline()
        
        pipeline.io("fetch", lambda: {"data": 1})
        assert len(pipeline.stages) == 1
        assert pipeline.stages[0].stage_type == "io"

    def test_method_chaining(self):
        """Test that methods return self for chaining."""
        pipeline = Pipeline()
        
        result = (pipeline
                  .stage("s1", lambda x: x)
                  .io("io1", lambda: 1)
                  .stage("s2", lambda x: x))
        
        assert result is pipeline
        assert len(pipeline.stages) == 3

    def test_through_membrane(self):
        """Test the through() method for adding types and hooks."""
        hook_calls = []
        
        def hook(result, ctx):
            hook_calls.append(result)
        
        pipeline = (Pipeline()
                    .stage("process", lambda x: x * 2)
                    .through(int, hooks=[hook]))
        
        stage = pipeline.stages[0]
        assert stage.output_type == int
        assert len(stage.hooks) == 1

    def test_through_shorthand(self):
        """Test the T() shorthand for through()."""
        pipeline = Pipeline().stage("test", lambda x: x).T(str)
        assert pipeline.stages[0].output_type == str

    def test_hook_method(self):
        """Test adding hooks with hook() method."""
        def my_hook(result, ctx):
            pass
        
        pipeline = Pipeline().stage("test", lambda x: x).hook(my_hook)
        assert len(pipeline.stages[0].hooks) == 1

    def test_sequential_execution(self):
        """Test basic sequential pipeline execution."""
        pipeline = (Pipeline()
                    .stage("add", lambda x: x + 10)
                    .stage("multiply", lambda x: x * 2)
                    .stage("stringify", lambda x: f"Result: {x}"))
        
        ctx = Ctx()
        ctx.set_stage_input("add", 5)
        result = pipeline.run(ctx)
        
        assert result == "Result: 30"
        assert ctx.get_stage_output("add") == 15
        assert ctx.get_stage_output("multiply") == 30
        assert ctx.get_stage_output("stringify") == "Result: 30"

    def test_pipeline_with_io(self):
        """Test pipeline with I/O operations."""
        data_store = {"value": 42}
        
        def fetch():
            return data_store["value"]
        
        def save(result):
            data_store["result"] = result
            return result
        
        pipeline = (Pipeline()
                    .io("fetch", fetch)
                    .stage("process", lambda x: x * 2)
                    .io("save", save))
        
        result = pipeline.run()
        assert result == 84
        assert data_store["result"] == 84

    def test_pipeline_with_hooks(self):
        """Test that hooks are executed during pipeline run."""
        hook_log = []
        
        def log_hook(result, ctx):
            hook_log.append(("log", result))
        
        def notify_hook(result, ctx):
            hook_log.append(("notify", result))
        
        pipeline = (Pipeline()
                    .stage("step1", lambda: 1)
                    .through(int, hooks=[log_hook])
                    .stage("step2", lambda x: x + 1)
                    .hook(notify_hook))
        
        result = pipeline.run()
        assert result == 2
        assert hook_log == [("log", 1), ("notify", 2)]

    def test_config_patterns_in_pipeline(self):
        """Test configuration access patterns in pipeline."""
        # Setup experiment with config
        class SweepExperiment:
            def __init__(self, cfg):
                self.cfg = cfg
            
            def load_data(self):
                # I/O operation that uses config
                return self.cfg["data_value"]
            
            def transform(self, data):
                # Stage that uses config
                return data * self.cfg["multiplier"]
        
        cfg = {"data_value": 10, "multiplier": 2}
        exp = SweepExperiment(cfg)
        
        # Build pipeline with method references
        pipeline = (Pipeline()
                    .io("load", exp.load_data)
                    .stage("transform", exp.transform)
                    .stage("add_constant", lambda x: x + 5))  # Lambda without config
        
        result = pipeline.run()
        assert result == 25  # (10 * 2) + 5

    def test_empty_pipeline(self):
        """Test that empty pipeline returns None."""
        pipeline = Pipeline()
        result = pipeline.run()
        assert result is None

    def test_build_validation(self):
        """Test that build validates stages."""
        # Stage validation now happens in __post_init__, not build()
        with pytest.raises(TypeError, match="func must be callable"):
            Stage(name="bad", func=None)

    def test_type_inference(self):
        """Test that type inference works for annotated functions."""
        def typed_func(x: int) -> str:
            return str(x)
        
        pipeline = Pipeline().stage("typed", typed_func)
        stage = pipeline.stages[0]
        
        # Type inference should pick up the annotations
        assert stage.input_type == int
        assert stage.output_type == str

    def test_type_inference_disabled(self):
        """Test that type inference can be disabled."""
        def typed_func(x: int) -> str:
            return str(x)
        
        pipeline = Pipeline().stage("typed", typed_func, infer_types=False)
        stage = pipeline.stages[0]
        
        # Types should not be inferred
        assert stage.input_type is None
        assert stage.output_type is None

    def test_context_flows_through_pipeline(self):
        """Test that context is preserved through execution."""
        def stage1(x):
            return x + 1
        
        def stage2(x):
            # Receives output from previous stage
            return x * 2
        
        pipeline = (Pipeline()
                    .stage("s1", stage1)
                    .stage("s2", stage2))
        
        ctx = Ctx()
        ctx.set_stage_input("s1", 10)
        result = pipeline.run(ctx)
        
        assert result == 22  # (10 + 1) * 2
        assert ctx.get_stage_output("s1") == 11
        assert ctx.get_stage_output("s2") == 22


class TestIntegration:
    """Integration tests for complete pipeline scenarios."""

    def test_data_processing_pipeline(self):
        """Test a realistic data processing pipeline."""
        # Simulate external data source
        external_data = [1, 2, 3, 4, 5]
        
        def load_data():
            return external_data
        
        def filter_evens(data):
            return [x for x in data if x % 2 == 0]
        
        def sum_values(data):
            return sum(data)
        
        def format_result(value):
            return {"sum": value, "count": 2}
        
        pipeline = (Pipeline()
                    .io("load", load_data)
                    .stage("filter", filter_evens)
                    .stage("sum", sum_values)
                    .stage("format", format_result))
        
        result = pipeline.run()
        assert result == {"sum": 6, "count": 2}  # 2 + 4 = 6

    def test_pipeline_with_mixed_operations(self):
        """Test pipeline with stages, I/O, and hooks."""
        events = []
        
        def log_event(result, ctx):
            events.append(f"Result: {result}")
        
        # Using method-based approach instead of context_aware
        class ConfigurableExperiment:
            def __init__(self, multiplier):
                self.cfg = {"multiplier": multiplier}
            
            def configure(self):
                return self.cfg
            
            def process(self, config):
                return config["multiplier"] * 10
        
        def save_result(value):
            # Simulate saving to database
            return {"saved": value}
        
        exp = ConfigurableExperiment(multiplier=3)
        
        pipeline = (Pipeline()
                    .stage("configure", exp.configure)
                    .through(dict, hooks=[log_event])
                    .stage("process", exp.process)
                    .through(int, hooks=[log_event])
                    .io("save", save_result))
        
        result = pipeline.run()
        
        assert result == {"saved": 30}
        assert events == ["Result: {'multiplier': 3}", "Result: 30"]
        
    def test_pipeline_without_context(self):
        """Test that pipeline creates its own context if not provided."""
        pipeline = Pipeline().stage("test", lambda: "hello")
        result = pipeline.run()
        assert result == "hello"