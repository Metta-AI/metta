"""Control flow primitives for tAXIOM pipelines."""

from __future__ import annotations

import concurrent.futures
from collections.abc import Callable
from typing import Any

from metta.sweep.axiom.core import Ctx, Pipeline, Stage


class ParallelStage(Stage):
    """Stage that executes multiple pipelines in parallel."""

    def __init__(self, name: str, pipelines: list[Pipeline]):
        """Initialize parallel stage.

        Args:
            name: Stage name
            pipelines: List of pipelines to execute in parallel
        """
        super().__init__(name=name, func=self._execute_parallel)
        self.pipelines = pipelines

    def _execute_parallel(self, ctx: Ctx) -> list[Any]:
        """Execute pipelines in parallel.

        Note: Each pipeline gets its own context copy to avoid shared state.
        """
        results = []

        # Use ThreadPoolExecutor for parallel execution
        # (Could use ProcessPoolExecutor for CPU-bound tasks)
        with concurrent.futures.ThreadPoolExecutor() as executor:
            # Create separate context for each pipeline
            futures = []
            for pipeline in self.pipelines:
                # Create independent context copy
                pipeline_ctx = Ctx()
                pipeline_ctx.metadata = ctx.metadata.copy()

                # Submit pipeline for execution
                future = executor.submit(pipeline.run, pipeline_ctx)
                futures.append(future)

            # Collect results
            for future in concurrent.futures.as_completed(futures):
                try:
                    result = future.result()
                    results.append(result)
                except Exception as e:
                    # Re-raise with context
                    raise RuntimeError(f"Parallel pipeline failed: {e}") from e

        return results


class WhileLoopStage(Stage):
    """Stage that executes a pipeline repeatedly while a condition is true."""

    def __init__(
        self, name: str, condition: Callable[[Ctx], bool], pipeline: Pipeline, max_iterations: int | None = None
    ):
        """Initialize while loop stage.

        Args:
            name: Stage name
            condition: Function that returns True to continue looping
            pipeline: Pipeline to execute in loop body
            max_iterations: Maximum iterations (safety limit)
        """
        super().__init__(name=name, func=self._execute_loop)
        self.condition = condition
        self.pipeline = pipeline
        self.max_iterations = max_iterations

    def _execute_loop(self, ctx: Ctx) -> Any:
        """Execute pipeline while condition is true."""
        iterations = 0
        last_result = None

        while self.condition(ctx):
            # Check iteration limit
            if self.max_iterations and iterations >= self.max_iterations:
                raise RuntimeError(f"While loop exceeded max iterations ({self.max_iterations})")

            # Execute pipeline with shared context
            last_result = self.pipeline.run(ctx)
            iterations += 1

        return last_result


class UntilLoopStage(Stage):
    """Stage that executes a pipeline repeatedly until a condition is true."""

    def __init__(
        self, name: str, condition: Callable[[Ctx], bool], pipeline: Pipeline, max_iterations: int | None = None
    ):
        """Initialize until loop stage.

        Args:
            name: Stage name
            condition: Function that returns True to stop looping
            pipeline: Pipeline to execute in loop body
            max_iterations: Maximum iterations (safety limit)
        """
        super().__init__(name=name, func=self._execute_loop)
        self.condition = condition
        self.pipeline = pipeline
        self.max_iterations = max_iterations

    def _execute_loop(self, ctx: Ctx) -> Any:
        """Execute pipeline until condition is true."""
        iterations = 0
        last_result = None

        while not self.condition(ctx):
            # Check iteration limit
            if self.max_iterations and iterations >= self.max_iterations:
                raise RuntimeError(f"Until loop exceeded max iterations ({self.max_iterations})")

            # Execute pipeline with shared context
            last_result = self.pipeline.run(ctx)
            iterations += 1

        return last_result


# Extension methods for Pipeline class


def add_control_flow_to_pipeline():
    """Add control flow methods to Pipeline class."""

    def parallel(self: Pipeline, *pipelines: Pipeline) -> Pipeline:
        """Execute multiple pipelines in parallel.

        Args:
            *pipelines: Pipelines to execute in parallel

        Returns:
            Self for method chaining
        """
        # Generate unique name for parallel stage
        parallel_name = f"parallel_{len(self.stages)}"

        # Create parallel stage
        parallel_stage = ParallelStage(parallel_name, list(pipelines))

        # Add to pipeline
        self.stages.append(parallel_stage)
        self.stage_names.add(parallel_name)

        return self

    def do_while(self: Pipeline, condition: Callable[[Ctx], bool]) -> Callable[[Pipeline], Pipeline]:
        """Execute a pipeline while a condition is true.

        Args:
            condition: Function that takes Ctx and returns bool

        Returns:
            Decorator that wraps a pipeline for repeated execution
        """

        def wrapper(inner_pipeline: Pipeline) -> Pipeline:
            # Generate unique name
            loop_name = f"while_loop_{len(self.stages)}"

            # Create while loop stage
            loop_stage = WhileLoopStage(loop_name, condition, inner_pipeline)

            # Add to pipeline
            self.stages.append(loop_stage)
            self.stage_names.add(loop_name)

            return self

        return wrapper

    def do_until(self: Pipeline, condition: Callable[[Ctx], bool]) -> Callable[[Pipeline], Pipeline]:
        """Execute a pipeline until a condition is true.

        Args:
            condition: Function that takes Ctx and returns bool

        Returns:
            Decorator that wraps a pipeline for repeated execution
        """

        def wrapper(inner_pipeline: Pipeline) -> Pipeline:
            # Generate unique name
            loop_name = f"until_loop_{len(self.stages)}"

            # Create until loop stage
            loop_stage = UntilLoopStage(loop_name, condition, inner_pipeline)

            # Add to pipeline
            self.stages.append(loop_stage)
            self.stage_names.add(loop_name)

            return self

        return wrapper

    def wait_until(self: Pipeline, condition: Callable[[Ctx], bool], check_interval: float = 1.0) -> Pipeline:
        """Wait until a condition becomes true.

        Args:
            condition: Function that takes Ctx and returns bool
            check_interval: Seconds between condition checks

        Returns:
            Self for method chaining
        """
        import time

        def wait_func(ctx: Ctx) -> None:
            while not condition(ctx):
                time.sleep(check_interval)

        return self.stage(f"wait_until_{len(self.stages)}", wait_func)

    # Monkey-patch methods onto Pipeline class
    Pipeline.parallel = parallel  # type: ignore
    Pipeline.do_while = do_while  # type: ignore
    Pipeline.do_until = do_until  # type: ignore
    Pipeline.wait_until = wait_until  # type: ignore


# Initialize control flow methods when module is imported
add_control_flow_to_pipeline()


# Helper functions for common control flow patterns


def map_over(pipeline: Pipeline, items_key: str, max_parallel: int | None = None) -> Callable[[Ctx], list[Any]]:
    """Create a function that maps a pipeline over items in context.

    Args:
        pipeline: Pipeline to apply to each item
        items_key: Key in context to get items from
        max_parallel: Maximum parallel executions

    Returns:
        Function that can be used as a stage
    """

    def map_func(ctx: Ctx) -> list[Any]:
        items = ctx.metadata.get(items_key, [])
        results = []

        if max_parallel and max_parallel > 1:
            # Parallel execution
            with concurrent.futures.ThreadPoolExecutor(max_workers=max_parallel) as executor:
                futures = []
                for item in items:
                    # Create context for item
                    item_ctx = Ctx()
                    item_ctx.metadata = {"item": item}
                    future = executor.submit(pipeline.run, item_ctx)
                    futures.append(future)

                # Collect results
                for future in concurrent.futures.as_completed(futures):
                    results.append(future.result())
        else:
            # Sequential execution
            for item in items:
                item_ctx = Ctx()
                item_ctx.metadata = {"item": item}
                result = pipeline.run(item_ctx)
                results.append(result)

        return results

    return map_func
