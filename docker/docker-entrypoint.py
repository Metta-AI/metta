#!/usr/bin/env python3

import json
import logging
import os
import sys
import traceback
from pathlib import Path
from typing import Any, Dict

from policy_evaluator import PolicyEvaluator

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    handlers=[logging.StreamHandler(sys.stderr)],
)
logger = logging.getLogger(__name__)


def get_required_env(key: str) -> str:
    value = os.environ.get(key)
    if not value:
        raise ValueError(f"Required environment variable {key} not set")
    return value


def get_optional_env(key: str, default: str = None) -> str | None:
    return os.environ.get(key, default)


def validate_environment() -> Dict[str, Any]:
    try:
        config = {
            "wandb_api_key": get_required_env("WANDB_API_KEY"),
            "policy_uri": get_required_env("POLICY_URI"),
            "wandb_entity": get_optional_env("WANDB_ENTITY", "metta-ai"),
            "wandb_project": get_optional_env("WANDB_PROJECT", "metta-research"),
            "simulation_suite": get_optional_env("SIMULATION_SUITE", "navigation"),
            "device": get_optional_env("DEVICE", "cpu"),
            "output_uri": get_optional_env("OUTPUT_URI"),
            "data_dir": get_optional_env("DATA_DIR", "/tmp/metta-data"),
            "run_name": get_optional_env("RUN_NAME"),
        }

        os.environ["WANDB_API_KEY"] = config["wandb_api_key"]

        logger.info("Environment validated successfully:")
        logger.info(f"  Policy URI: {config['policy_uri']}")
        logger.info(f"  WandB Entity: {config['wandb_entity']}")
        logger.info(f"  WandB Project: {config['wandb_project']}")
        logger.info(f"  Simulation Suite: {config['simulation_suite']}")
        logger.info(f"  Device: {config['device']}")
        logger.info(f"  Data Dir: {config['data_dir']}")

        return config

    except Exception as e:
        logger.error(f"Environment validation failed: {e}")
        raise


def main():
    try:
        logger.info("Starting policy evaluation service...")

        config = validate_environment()

        data_dir = Path(config["data_dir"])
        data_dir.mkdir(parents=True, exist_ok=True)

        evaluator = PolicyEvaluator(config)

        logger.info(f"Evaluating policy: {config['policy_uri']}")
        results = evaluator.evaluate_policy()

        logger.info("Policy evaluation completed successfully")

        sys.stderr.flush()
        sys.stdout.flush()

        print("===JSON_OUTPUT_START===")
        print(json.dumps(results, indent=2))
        print("===JSON_OUTPUT_END===")

        return 0

    except Exception as e:
        logger.error(f"Policy evaluation failed: {e}")
        logger.error(f"Traceback: {traceback.format_exc()}")

        error_result = {
            "error": str(e),
            "traceback": traceback.format_exc(),
            "policy_uri": config.get("policy_uri", "unknown") if "config" in locals() else "unknown",
        }

        print("===JSON_OUTPUT_START===")
        print(json.dumps(error_result, indent=2))
        print("===JSON_OUTPUT_END===")

        return 1


if __name__ == "__main__":
    sys.exit(main())
