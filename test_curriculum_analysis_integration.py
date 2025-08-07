#!/usr/bin/env python3
"""
Minimal integration test for curriculum analysis with real curricula.

This test verifies that the curriculum analysis framework works with
real curricula from the main codebase.
"""

import logging
import sys
from pathlib import Path

import hydra
from omegaconf import DictConfig

# Add the metta directory to the path
sys.path.insert(0, str(Path(__file__).parent))


# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@hydra.main(version_base=None, config_path="configs", config_name="trainer/test_curriculum_analysis")
def test_curriculum_analysis_integration(cfg: DictConfig) -> None:
    """Test curriculum analysis integration with real curricula."""
    logger.info("Testing curriculum analysis integration...")

    try:
        # Create trainer config - cfg has a trainer section
        trainer_cfg = cfg.trainer

        # Set analysis mode
        trainer_cfg.analysis_mode = True
        # Use configuration values instead of overriding
        # trainer_cfg.analysis_epochs = 3  # Very small for testing
        # trainer_cfg.analysis_tasks_per_epoch = 2
        trainer_cfg.analysis_output_dir = "integration_test"

        logger.info(f"Testing with curriculum: {trainer_cfg.curriculum}")
        logger.info(f"Analysis mode: {trainer_cfg.analysis_mode}")

        # Import and run analysis
        from metta.mettagrid.curriculum.util import curriculum_from_config_path
        from metta.rl.curriculum_analysis import run_curriculum_analysis

        # Load curriculum
        curriculum = curriculum_from_config_path(trainer_cfg.curriculum, trainer_cfg.env_overrides)

        # Run analysis
        results = run_curriculum_analysis(trainer_cfg=trainer_cfg, curriculum=curriculum, oracle_curriculum=None)

        logger.info("Integration test completed successfully!")
        logger.info(f"Results summary: {results['summary']}")

        # Check output files
        output_dir = Path(trainer_cfg.analysis_output_dir)
        if output_dir.exists():
            files = list(output_dir.glob("*.csv"))
            logger.info(f"Generated files: {[f.name for f in files]}")

        return results

    except Exception as e:
        logger.error(f"Integration test failed: {e}")
        import traceback

        traceback.print_exc()
        raise


if __name__ == "__main__":
    test_curriculum_analysis_integration()
