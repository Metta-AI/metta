# Component-Based Architecture Refactoring Summary

## Overview

This refactoring transforms the Metta codebase from a functional/procedural approach to a component-based architecture while maintaining the low-level nature of the components. The refactoring introduces several manager classes that encapsulate specific responsibilities in the training workflow.

## New Components

### 1. **RolloutManager** (`metta/rl/components/rollout_manager.py`)
- Handles the rollout phase of training
- Manages experience collection from the environment
- Encapsulates the interaction loop between agent and environment

### 2. **TrainingManager** (`metta/rl/components/training_manager.py`)
- Manages the PPO training phase
- Handles advantage computation and minibatch updates
- Manages the training loop including early stopping based on KL divergence

### 3. **StatsManager** (`metta/rl/components/stats_manager.py`)
- Consolidates all statistics tracking and processing
- Manages rollout stats, gradient stats, and weight stats
- Handles building stats dictionaries for logging
- Maintains evaluation scores

### 4. **EnvironmentManager** (`metta/rl/components/environment_manager.py`)
- Manages environment creation and configuration
- Handles batch size calculations
- Provides access to curriculum and environment properties
- Manages environment lifecycle

### 5. **OptimizerManager** (`metta/rl/components/optimizer_manager.py`)
- Handles optimizer creation and configuration
- Manages optimizer state loading from checkpoints
- Provides learning rate management utilities

### 6. **EvaluationManager** (`metta/rl/components/evaluation_manager.py`)
- Manages policy evaluation during training
- Handles replay generation
- Tracks evaluation scores and statistics
- Manages simulation suite execution

### 7. **Trainer** (`metta/rl/components/trainer.py`)
- Main orchestrator class that coordinates all components
- Manages the complete training lifecycle
- Handles setup, training loop, checkpointing, and cleanup
- Provides a clean interface for training execution

## Key Design Decisions

1. **Low-Level Components**: Each component remains focused and low-level, handling specific responsibilities without becoming monolithic.

2. **State Management**: Components manage their own state while the Trainer class coordinates the overall workflow.

3. **Backward Compatibility**: The refactoring maintains compatibility with existing interfaces through adapter functions.

4. **Separation of Concerns**: Each component has a clear, single responsibility making the code more maintainable and testable.

## Usage Examples

### Simple Training Script
```python
from metta.rl.components import Trainer

# Create trainer
trainer = Trainer(
    trainer_config=trainer_config,
    run_dir=run_dir,
    run_name=run_name,
    checkpoint_dir=checkpoint_dir,
    replay_dir=replay_dir,
    stats_dir=stats_dir,
    wandb_config=wandb_config,
    global_config=global_config,
)

try:
    # Set up components
    trainer.setup(vectorization="multiprocessing")
    
    # Run training
    trainer.train()
    
finally:
    # Clean up
    trainer.cleanup()
```

### Integration with Existing Code
The new `trainer_component.py` provides a drop-in replacement for the functional `train` interface, allowing gradual migration of existing code.

## Benefits

1. **Modularity**: Each component can be tested and modified independently
2. **Reusability**: Components can be reused in different contexts
3. **Clarity**: The code structure better reflects the logical organization of the training workflow
4. **Extensibility**: New features can be added by extending components or adding new ones
5. **Maintainability**: Bugs are easier to isolate and fix within specific components

## Migration Path

1. New code should use the component-based `Trainer` class directly
2. Existing code can continue using the functional interface through `trainer_component.py`
3. Gradually migrate existing scripts to use the new components directly
4. The original functional code remains available for reference and backward compatibility