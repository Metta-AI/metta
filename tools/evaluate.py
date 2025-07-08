#!/usr/bin/env -S uv run
"""
Evaluate a trained policy from Weights & Biases or local file.

What it does:
- Loads a policy from a checkpoint URI (file://, wandb://, etc.)
- Runs simulation episodes
- Stores metrics in both local DuckDB and remote PostgreSQL stats server
- Generates replay files for visualization in MetaScope
- Optionally uploads results to wandb


Adds rows to stats server accordingly:
- **policies**: Reused if already exists (by URL)
- **episodes**: Created with `stats_epoch=NULL` for standalone eval
- **episode_agent_metrics**: All agent metrics per episode

How it works:
1. Fetches the policy using PolicyStore
2. Runs episodes with SimulationSuite
3. Stats collection:
   - Local DuckDB stores episode data during simulation
   - Syncs to remote stats server if configured
   - Reuses existing policy entries (no duplication)
4. Scoring: Uses `get_average_metric_by_filter` for normalized metrics
5. Output: Returns scores and paths to stats/replay files

Example usage:
./tools/evaluate.py policy_uri=wandb://entity/project/model:version
./tools/evaluate.py policy_uri=wandb://project/artifact:version wandb=metta-research
./tools/evaluate.py policy_uri=file://./path/to/checkpoint.pt sim=navigation
./tools/evaluate.py policy_uri=file://./path/to/checkpoint.pt run=custom-name upload_to_wandb=false
"""

import sys
from datetime import datetime
from pathlib import Path

import hydra
from omegaconf import DictConfig, OmegaConf

from metta.common.util.script_decorators import get_metta_logger, metta_script
from metta.eval.policy_evaluator import EvaluateJobConfig, evaluate_policy


def _determine_run_name(policy_uri: str) -> str:
    if policy_uri.startswith("file://"):
        # Extract checkpoint name from file path
        checkpoint_path = Path(policy_uri.replace("file://", ""))
        return f"eval_{checkpoint_path.stem}"
    elif policy_uri.startswith("wandb://"):
        # Extract artifact name from wandb URI
        # Format: wandb://entity/project/artifact:version
        artifact_part = policy_uri.split("/")[-1]
        return f"eval_{artifact_part.replace(':', '_')}"
    else:
        # Fallback to timestamp
        return f"eval_{datetime.now().strftime('%Y%m%d_%H%M%S')}"


@hydra.main(config_path="../configs", config_name="evaluate_job", version_base=None)
@metta_script
def main(cfg: DictConfig) -> int:
    logger = get_metta_logger()
    if not cfg.get("run"):
        cfg.run = _determine_run_name(cfg.get("policy_uri"))
        logger.info(f"Auto-generated run name: {cfg.run}")

    dict_cfg = OmegaConf.to_container(cfg, resolve=True)
    assert isinstance(dict_cfg, dict), "cfg must be a DictConfig"

    logger.info(f"Evaluating policy: {dict_cfg}")
    config = EvaluateJobConfig(dict_cfg)
    logger.info("Evaluating policy:\n" + config.model_dump_json(indent=2))
    results = evaluate_policy(config, logger)
    logger.info("Evaluation complete!:\n" + results.model_dump_json(indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
