#!/usr/bin/env uv run
"""A/B testing with typed states and clean separation."""

from typing import Optional

from pydantic import Field

from metta.sweep.axiom import Context, Pipeline, PipelineState


class TrainingState(PipelineState):
    """Strongly-typed training state - immediately shows data flow."""
    # Data configuration
    dataset: str = ""
    epochs: int = 0
    
    # Training configuration
    optimizer_type: str = "unknown"
    learning_rate: float = 0.001
    
    # Results
    model: Optional[str] = None
    accuracy: Optional[float] = None
    evaluated: bool = False


def load_data(state: TrainingState, ctx: Context) -> None:
    """Load training data."""
    state.dataset = "mnist"
    state.epochs = 10
    print(f"Loading {state.dataset} for {state.epochs} epochs")


def setup_optimizer_default(state: TrainingState, ctx: Context) -> None:
    """Default optimizer setup."""
    state.optimizer_type = "SGD"
    state.learning_rate = 0.01


def train_model(state: TrainingState, ctx: Context) -> None:
    """Train the model with configured optimizer."""
    print(f"Training with {state.optimizer_type} (lr={state.learning_rate})")
    state.model = f"model_trained_with_{state.optimizer_type}"
    # Simulate accuracy based on optimizer
    state.accuracy = 0.85 if state.optimizer_type == "SGD" else 0.92


def evaluate_default(state: TrainingState, ctx: Context) -> None:
    """Default evaluation."""
    print("Running basic evaluation")
    state.evaluated = True


def save_model(state: TrainingState, ctx: Context) -> None:
    """Save the trained model."""
    print(f"Saving {state.model} (accuracy: {state.accuracy})")


def base_pipeline() -> Pipeline:
    """Base training pipeline with variation points."""
    state = TrainingState()
    
    return (
        Pipeline(state)
        .stage("load", load_data)
        .join("optimizer", setup_optimizer_default)  # Variation point
        .stage("train", train_model)
        .join("evaluation", evaluate_default)  # Variation point
        .stage("save", save_model)
    )


def variant_a() -> Pipeline:
    """Variant A: Adam optimizer with thorough evaluation."""
    pipeline = base_pipeline()
    
    # Override optimizer with Adam
    def setup_adam(state: TrainingState, ctx: Context) -> None:
        state.optimizer_type = "Adam"
        state.learning_rate = 0.001
        print(f"Configuring Adam optimizer at stage {ctx.get('current_stage')}")
    
    pipeline.override("optimizer", setup_adam)
    
    # Override with thorough evaluation
    def thorough_eval(state: TrainingState, ctx: Context) -> None:
        print("Running thorough evaluation with cross-validation")
        state.evaluated = True
        state.metadata['cv_score'] = 0.95
        state.metadata['eval_stages'] = len(ctx.get('stage_history', []))
    
    pipeline.override("evaluation", thorough_eval)
    
    return pipeline


def variant_b() -> Pipeline:
    """Variant B: RMSprop optimizer, minimal evaluation."""
    pipeline = base_pipeline()
    
    # Override with RMSprop
    def setup_rmsprop(state: TrainingState, ctx: Context) -> None:
        state.optimizer_type = "RMSprop"
        state.learning_rate = 0.0001
    
    pipeline.override("optimizer", setup_rmsprop)
    
    # Skip evaluation
    pipeline.override("evaluation", lambda s, c: None)
    
    return pipeline


def main():
    """Run both variants and compare typed states."""
    print("=" * 50)
    print("Variant A: Adam with thorough evaluation")
    print("=" * 50)
    state_a = variant_a().run()
    
    print("\n" + "=" * 50)
    print("Variant B: RMSprop with minimal evaluation")
    print("=" * 50)
    state_b = variant_b().run()
    
    print("\n" + "=" * 50)
    print("Results comparison (typed state makes this clear!):")
    print(f"Variant A:")
    print(f"  Model: {state_a.model}")
    print(f"  Accuracy: {state_a.accuracy}")
    print(f"  Evaluated: {state_a.evaluated}")
    print(f"  CV Score: {state_a.metadata.get('cv_score')}")
    
    print(f"\nVariant B:")
    print(f"  Model: {state_b.model}")
    print(f"  Accuracy: {state_b.accuracy}")
    print(f"  Evaluated: {state_b.evaluated}")


if __name__ == "__main__":
    main()