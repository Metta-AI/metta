import logging
from typing import Any, Dict

from omegaconf import DictConfig, OmegaConf

from metta.common.util.numpy_helpers import clean_numpy_types
from metta.common.util.retry import retry_on_exception
from metta.sweep.protein_metta import MettaProtein

logger = logging.getLogger(__name__)


@retry_on_exception(max_retries=10, retry_delay=0.1, exceptions=(ValueError,))
def generate_protein_suggestion(trainer_config: Dict[str, Any], protein: MettaProtein) -> Dict[str, Any]:
    """Generate a protein suggestion.
    We only validate constraints related batch_size, minibatch_size, bppt.
    We must have: minibatch_size divides batch_size, bppt divides minibatch_size.
    We do not validate other constraints, such as total_timesteps >= batch_size * bppt, etc...

    Args:
        config: Configuration containing trainer settings
        protein: MettaProtein instance

    Returns:
        Cleaned protein suggestion dictionary
    """
    suggestion, _ = protein.suggest()
    logger.info(f"Protein suggestion: {suggestion}")

    # Extract trainer config for validation
    try:
        validate_protein_suggestion(trainer_config, suggestion)
    except Exception as e:
        # Catch the invalid exception and record it so Protein can learn from it
        logger.warning(f"Invalid suggestion: {e}")
        protein.observe_failure(suggestion)
        # TODO: Add protein observation to wandb?
        raise e
    return clean_numpy_types(suggestion)


# TODO: Move this out of here, and creating a WandB Utils file.
def validate_protein_suggestion(trainer_config: Dict[str, Any], suggestion: Dict[str, Any]) -> None:
    """Validate a protein suggestion.
    We only validate constraints related total_timesteps, batch_size, minibatch_size, bppt.
    We must have: minibatch_size divides batch_size, bppt divides minibatch_size.

    Args:
        trainer_config: Trainer configuration dictionary
        suggestion: The suggestion to validate
    """
    # Get base values from trainer config
    batch_size = trainer_config.get("batch_size")
    minibatch_size = trainer_config.get("minibatch_size")
    bppt = trainer_config.get("bptt_horizon")

    # Parse the protein suggestion
    if "trainer" in suggestion:
        if "batch_size" in suggestion["trainer"]:
            batch_size = suggestion["trainer"]["batch_size"]
        if "minibatch_size" in suggestion["trainer"]:
            minibatch_size = suggestion["trainer"]["minibatch_size"]
        if "bptt_horizon" in suggestion["trainer"]:
            bppt = suggestion["trainer"]["bptt_horizon"]

    # Validate the suggestion
    if batch_size is not None and minibatch_size is not None and batch_size % minibatch_size != 0:
        raise ValueError(f"Batch size {batch_size} must be divisible by minibatch size {minibatch_size}")
    if minibatch_size is not None and bppt is not None and minibatch_size % bppt != 0:
        raise ValueError(f"Minibatch size {minibatch_size} must be divisible by bppt {bppt}")


# TODO: This is the only function that uses OmegaConf, which I don't like.
# It should maybe live somewhere else.
def apply_protein_suggestion(config: DictConfig, suggestion: dict):
    """Apply suggestions to a configuration object using deep merge.

    Args:
        config: The configuration object to modify (must be a DictConfig)
        suggestion: The suggestions to apply (cleaned dict)
    """
    for key, value in suggestion.items():
        if key == "suggestion_uuid":
            continue

        # Clean numpy types from the value before applying
        cleaned_value = clean_numpy_types(value)

        # For nested structures, merge instead of overwrite
        if key in config and isinstance(config[key], DictConfig) and isinstance(cleaned_value, dict):
            config[key] = OmegaConf.merge(config[key], cleaned_value)
        else:
            config[key] = cleaned_value
