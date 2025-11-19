"""Protein-based Bayesian optimization sweep using the SweepOrchestrator framework."""

import logging
from typing import Optional

from metta.sweep.config import SweepOrchestratorConfig
from metta.sweep.models import JobDefinition
from metta.sweep.optimizer.protein import ProteinOptimizer
from metta.sweep.orchestrator import SweepOrchestrator, Trial, TrialState
from metta.sweep.protein_config import ProteinConfig
from metta.sweep.protocols import Dispatcher, Store
from metta.sweep.utils import create_eval_job, create_training_job, generate_run_id

logger = logging.getLogger(__name__)


class ProteinSweep(SweepOrchestrator):
    """Bayesian optimization sweep using the Protein optimizer.

    Simple implementation that:
    1. Collects observations from completed trials
    2. Passes them to Protein optimizer
    3. Gets suggestions and creates trials
    """

    def __init__(
        self,
        experiment_id: str,
        dispatcher: Dispatcher,
        store: Store,
        sweep_config: SweepOrchestratorConfig,
        protein_config: ProteinConfig,
        recipe_module: str,
        train_entrypoint: str = "train",
        eval_entrypoint: str = "evaluate",
        max_trials: int = 50,
        batch_size: int = 4,
        gpus: int = 1,
        nodes: int = 1,
        train_overrides: Optional[dict] = None,
        eval_overrides: Optional[dict] = None,
    ):
        """Initialize Protein sweep.

        Args:
            experiment_id: Unique experiment identifier
            dispatcher: Job dispatcher
            store: Data store (e.g., WandB)
            sweep_config: Orchestrator configuration
            protein_config: Protein optimizer configuration
            recipe_module: Module containing train/eval functions
            train_entrypoint: Name of training function
            eval_entrypoint: Name of evaluation function
            max_trials: Maximum number of trials to run
            batch_size: Number of parallel suggestions
            gpus: GPUs per job
            nodes: Nodes per job
            train_overrides: Additional training config overrides
            eval_overrides: Additional evaluation config overrides
        """
        # Set attributes before calling super().__init__ since parent calls setup()
        self.protein_config = protein_config
        self.recipe_module = recipe_module
        self.train_entrypoint = train_entrypoint
        self.eval_entrypoint = eval_entrypoint
        self.max_trials = max_trials
        self.batch_size = batch_size
        self.gpus = gpus
        self.nodes = nodes
        self.train_overrides = train_overrides or {}
        self.eval_overrides = eval_overrides or {}
        self.optimizer = None

        # Track trial number for unique IDs
        self.trial_counter = 0

        # Now call parent init which will call setup()
        super().__init__(experiment_id, dispatcher, store, sweep_config)

    def setup(self) -> None:
        """Initialize the Protein optimizer."""

        self.optimizer = ProteinOptimizer(self.protein_config)
        logger.info(f"Initialized Protein optimizer for metric: {self.protein_config.metric}")

        # If resuming, update trial counter based on existing trials
        if self.resume and self.trials:
            # Extract the highest trial number from existing trial IDs
            for trial_id in self.trials.keys():
                if trial_id.startswith(f"{self.experiment_id}_trial_"):
                    try:
                        # Handle both formats: "exp_trial_14" and "exp_trial_14_abc"
                        trial_part = trial_id.split("_trial_")[-1]
                        # Split by underscore to get the number part (before hash if present)
                        trial_num_str = trial_part.split("_")[0]
                        trial_num = int(trial_num_str)
                        self.trial_counter = max(self.trial_counter, trial_num)
                    except (ValueError, IndexError):
                        pass
            logger.info(f"Resuming from trial counter: {self.trial_counter}")

    def suggest_trials(self, n_slots: int) -> list[Trial]:
        """Generate new trial suggestions using Protein.

        Args:
            n_slots: Number of available resource slots

        Returns:
            List of trials to run
        """
        # Collect observations from completed trials
        observations = []
        for trial_id, obs in self.observations.items():
            trial = self.trials[trial_id]
            # Create observation in format Protein expects
            observation = {
                "suggestion": trial.params,  # The parameters that were tried
                "score": obs.score,
                "cost": obs.cost,
            }
            observations.append(observation)

        # Request suggestions from optimizer
        n_suggestions = min(n_slots, self.batch_size)
        logger.info(f"Requesting {n_suggestions} suggestions with {len(observations)} observations")

        suggestions = self.optimizer.suggest(observations, n_suggestions=n_suggestions)

        # Convert suggestions to trials
        trials = []
        for suggestion in suggestions:
            self.trial_counter += 1
            trial_id = generate_run_id(self.experiment_id, self.trial_counter)

            trial = Trial(
                id=trial_id,
                params=suggestion,  # Suggestion is the parameter dict
            )
            trials.append(trial)

        return trials

    def should_stop(self) -> bool:
        """Determine if the sweep should terminate."""
        return self.n_completed >= self.max_trials

    def _trial_to_job(self, trial: Trial) -> JobDefinition:
        """Convert a trial to a JobDefinition for the dispatcher.

        Override the base implementation to create proper JobDefinition
        with recipe information and hyperparameters as overrides.
        Uses the helper functions from adaptive.utils for consistency.

        Args:
            trial: Trial to convert

        Returns:
            JobDefinition suitable for dispatcher
        """
        # Determine if this is for evaluation based on trial state
        # When dispatching evaluation, trial is in AWAITING_EVALUATION state
        if trial.state == TrialState.AWAITING_EVALUATION:
            # Create evaluation job using helper
            job = create_eval_job(
                run_id=trial.id,
                experiment_id=self.experiment_id,
                recipe_module=self.recipe_module,
                eval_entrypoint=self.eval_entrypoint,
                eval_overrides=self.eval_overrides,
            )
        else:
            # Create training job using helper
            # Merge trial parameters (hyperparameters) with training overrides
            all_overrides = {**self.train_overrides, **trial.params}

            job = create_training_job(
                run_id=trial.id,
                experiment_id=self.experiment_id,
                recipe_module=self.recipe_module,
                train_entrypoint=self.train_entrypoint,
                gpus=self.gpus,
                nodes=self.nodes,
                train_overrides=all_overrides,
            )

        return job
