#!/usr/bin/env uv run
"""Basic pipeline example - clean State/Context separation."""

from pydantic import Field

from metta.sweep.axiom import Pipeline, PipelineState, Context


class GreetingState(PipelineState):
    """Typed state for greeting pipeline - shows exactly what flows through."""
    message: str = ""
    processed: bool = False
    save_count: int = 0


def greet(state: GreetingState, ctx: Context) -> None:
    """Set greeting message - modifies state, reads context."""
    state.message = "Hello from Axiom!"
    print(f"Stage '{ctx.get('current_stage')}' setting message")


def process(state: GreetingState, ctx: Context) -> None:
    """Process the message."""
    state.processed = True
    state.metadata['processing_stage'] = ctx.get('current_stage')


def save(state: GreetingState, ctx: Context) -> None:
    """Save operation - I/O stage."""
    state.save_count += 1
    print(f"Saving (attempt {state.save_count}): {state.message}")
    print(f"  Processed: {state.processed}")
    print(f"  Pipeline path: {ctx.get('pipeline_path')}")


def main():
    """Demonstrate clean State/Context separation."""
    # Create typed state - immediately shows data flow
    state = GreetingState()
    
    # Build pipeline with clear stage signatures
    pipeline = (
        Pipeline(state)
        .stage("greet", greet)
        .stage("process", process)
        .io("save", save)
    )
    
    # Run it - returns the mutated state
    final_state = pipeline.run()
    
    print(f"\nFinal state:")
    print(f"  Message: {final_state.message}")
    print(f"  Processed: {final_state.processed}")
    print(f"  Metadata: {final_state.metadata}")


if __name__ == "__main__":
    main()