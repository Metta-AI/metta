import json
import logging
import subprocess
import sys
import tempfile
from datetime import datetime
from pathlib import Path
from typing import Any, Dict

import wandb
from omegaconf import OmegaConf

logger = logging.getLogger(__name__)


class PolicyEvaluator:
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.data_dir = Path(config["data_dir"])
        self.run_name = config.get("run_name") or self._generate_run_name()

        wandb.init(
            project=config["wandb_project"],
            entity=config["wandb_entity"],
            job_type="policy_evaluation",
            name=self.run_name,
            config=config,
        )

        logger.info(f"Initialized PolicyEvaluator with run name: {self.run_name}")

    def _generate_run_name(self) -> str:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        policy_suffix = self._extract_policy_name(self.config["policy_uri"])
        return f"eval_{policy_suffix}_{timestamp}"

    def _extract_policy_name(self, policy_uri: str) -> str:
        if policy_uri.startswith("wandb://"):
            return policy_uri.split("/")[-1].replace(":", "_")
        elif policy_uri.startswith("file://"):
            return Path(policy_uri.replace("file://", "")).stem
        else:
            return "unknown"

    def _create_sim_config(self) -> Dict[str, Any]:
        run_dir = self.data_dir / "runs" / self.run_name
        run_dir.mkdir(parents=True, exist_ok=True)

        return {
            "run": self.run_name,
            "policy_uri": self.config["policy_uri"],
            "device": self.config["device"],
            "data_dir": str(self.data_dir),
            "run_dir": str(run_dir),
            "sim_job": {
                "policy_uris": [self.config["policy_uri"]],
                "simulation_suite": {"name": self.config["simulation_suite"]},
                "stats_dir": str(run_dir / "stats"),
                "stats_db_uri": str(run_dir / "stats.db"),
                "replay_dir": str(run_dir / "replays" / "evals"),
                "selector_type": "top",
            },
            "wandb": {"project": self.config["wandb_project"], "entity": self.config["wandb_entity"], "mode": "online"},
        }

    def _write_config_file(self, config: Dict[str, Any]) -> Path:
        config_file = self.data_dir / "temp_config.yaml"
        with open(config_file, "w") as f:
            yaml_content = OmegaConf.to_yaml(OmegaConf.create(config))
            f.write(yaml_content)
        return config_file

    def evaluate_policy(self) -> Dict[str, Any]:
        try:
            logger.info("Starting policy evaluation...")

            sim_config = self._create_sim_config()
            config_file = self._write_config_file(sim_config)

            logger.info(f"Using config file: {config_file}")
            logger.info(f"Config: {json.dumps(sim_config, indent=2)}")

            sim_script = Path(__file__).parent.parent / "tools" / "sim.py"
            if not sim_script.exists():
                raise FileNotFoundError(f"sim.py not found at {sim_script}")

            cmd = [
                sys.executable,
                str(sim_script),
                f"--config-path={config_file.parent}",
                f"--config-name={config_file.stem}",
            ]

            logger.info(f"Running command: {' '.join(cmd)}")

            result = subprocess.run(
                cmd, capture_output=True, text=True, cwd=str(sim_script.parent.parent), timeout=3600
            )

            if result.returncode != 0:
                logger.error(f"sim.py failed with return code {result.returncode}")
                logger.error(f"stdout: {result.stdout}")
                logger.error(f"stderr: {result.stderr}")
                raise RuntimeError(f"Policy evaluation failed: {result.stderr}")

            logger.info("sim.py completed successfully")
            logger.info(f"stdout: {result.stdout}")

            output = self._parse_sim_output(result.stdout)

            if self.config.get("output_uri"):
                self._upload_results(output)

            wandb.log(
                {
                    "evaluation_status": "success",
                    "policy_uri": self.config["policy_uri"],
                    "simulation_suite": self.config["simulation_suite"],
                }
            )

            return output

        except Exception as e:
            logger.error(f"Policy evaluation failed: {e}")
            wandb.log({"evaluation_status": "failed", "error": str(e), "policy_uri": self.config["policy_uri"]})
            raise
        finally:
            if "config_file" in locals():
                config_file.unlink(missing_ok=True)
            wandb.finish()

    def _parse_sim_output(self, stdout: str) -> Dict[str, Any]:
        try:
            lines = stdout.strip().split("\n")
            json_start = None
            json_end = None

            for i, line in enumerate(lines):
                if line.strip() == "===JSON_OUTPUT_START===":
                    json_start = i + 1
                elif line.strip() == "===JSON_OUTPUT_END===":
                    json_end = i
                    break

            if json_start is None or json_end is None:
                logger.warning("No JSON output markers found, attempting to parse entire output")
                return {"raw_output": stdout, "parsed": False}

            json_content = "\n".join(lines[json_start:json_end])
            return json.loads(json_content)

        except json.JSONDecodeError as e:
            logger.error(f"Failed to parse JSON output: {e}")
            return {"raw_output": stdout, "parse_error": str(e), "parsed": False}

    def _upload_results(self, results: Dict[str, Any]):
        try:
            output_uri = self.config["output_uri"]

            if output_uri.startswith("wandb://"):
                artifact = wandb.Artifact(name=f"eval_results_{self.run_name}", type="evaluation_results")

                with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
                    json.dump(results, f, indent=2)
                    temp_file = f.name

                artifact.add_file(temp_file, name="results.json")
                wandb.log_artifact(artifact)

                Path(temp_file).unlink()
                logger.info(f"Results uploaded to WandB artifact: {artifact.name}")

            elif output_uri.startswith("s3://"):
                logger.warning("S3 upload not implemented yet")

            else:
                output_path = Path(output_uri)
                output_path.parent.mkdir(parents=True, exist_ok=True)
                with open(output_path, "w") as f:
                    json.dump(results, f, indent=2)
                logger.info(f"Results written to: {output_path}")

        except Exception as e:
            logger.error(f"Failed to upload results: {e}")
            raise
