"""Evaluation tool for Gymnasium environments using StableBaselines3."""

import contextlib
import shutil
import tempfile
from pathlib import Path
from typing import Any, Optional

import boto3
from botocore.exceptions import ClientError
from pydantic import Field
from stable_baselines3 import PPO, A2C, SAC
from stable_baselines3.common.evaluation import evaluate_policy
from stable_baselines3.common.env_util import make_vec_env

from metta.common.tool import Tool
from metta.common.util.log_config import getRankAwareLogger
from metta.common.wandb.context import WandbConfig, WandbContext
from metta.tools.utils.auto_config import auto_run_name, auto_wandb_config
from metta.utils.file import local_copy
from metta.utils.uri import ParsedURI

logger = getRankAwareLogger(__name__)


class EvaluateSB3Tool(Tool):
    """Tool for evaluating trained StableBaselines3 models on Gymnasium environments.

    This tool provides a CLI interface for evaluating trained models
    on standard RL benchmarks.

    Example usage:
        # From recipe
        def evaluate() -> EvaluateSB3Tool:
            return EvaluateSB3Tool(
                env_id="LunarLander-v3",
                model_path="./models/lunarlander_ppo.zip",
            )

        # CLI invocation
        uv run ./tools/run.py module.evaluate model_path=./trained_model.zip
    """

    # Run configuration
    run: Optional[str] = Field(
        default=None,
        description="Run name for organizing evaluation experiments"
    )

    # Environment configuration
    env_id: str = Field(
        default="LunarLander-v3",
        description="Gymnasium environment ID"
    )

    # Model configuration
    model_path: Optional[str] = Field(
        default=None,
        description="Path to the trained model (.zip file or directory)"
    )

    algorithm: str = Field(
        default="PPO",
        description="Algorithm used to train the model (PPO, A2C, or SAC)"
    )

    # Evaluation configuration
    n_eval_episodes: int = Field(
        default=100,
        description="Number of episodes to evaluate"
    )

    deterministic: bool = Field(
        default=True,
        description="Use deterministic actions during evaluation"
    )

    render: bool = Field(
        default=False,
        description="Render the environment during evaluation"
    )

    verbose: int = Field(
        default=1,
        description="Verbosity level (0=silent, 1=info, 2=debug)"
    )

    # Success threshold (environment-specific)
    success_threshold: Optional[float] = Field(
        default=None,
        description="Threshold for considering an episode successful (env-specific)"
    )

    # WandB configuration
    wandb: WandbConfig = Field(
        default_factory=WandbConfig.Unconfigured,
        description="WandB configuration"
    )
    group: Optional[str] = Field(
        default=None,
        description="WandB group for organizing runs"
    )
    tags: list[str] = Field(
        default_factory=list,
        description="Tags for the run"
    )
    policy_uri: Optional[str] = Field(
        default=None,
        description="URI to fetch the trained model from (supports s3:// and file://)"
    )
    push_metrics_to_wandb: bool = Field(
        default=False,
        description="Whether to push evaluation metrics back to WandB"
    )
    stats_dir: Optional[str] = Field(
        default=None,
        description="Directory to store evaluation stats (unused, for sweep compatibility)"
    )
    enable_replays: bool = Field(
        default=False,
        description="Enable replay generation (unused, for sweep compatibility)"
    )
    stats_server_uri: Optional[str] = Field(
        default=None,
        description="Stats server URI (unused, for sweep compatibility)"
    )
    def invoke(self, args: dict[str, Any]) -> int | None:
        """Execute the evaluation tool.

        Args:
            args: Command-line arguments

        Returns:
            Exit code (0 for success, non-zero for failure)
        """
        # Override model_path if provided in args
        if "model_path" in args:
            self.model_path = args["model_path"]

        if "policy_uri" in args:
            self.policy_uri = args["policy_uri"]

        if "push_metrics_to_wandb" in args:
            self.push_metrics_to_wandb = args["push_metrics_to_wandb"]

        if "stats_dir" in args:
            self.stats_dir = args["stats_dir"]

        if "enable_replays" in args:
            self.enable_replays = args["enable_replays"]

        if "stats_server_uri" in args:
            self.stats_server_uri = args["stats_server_uri"]

        # Handle run name from args
        if "run" in args:
            if self.run is None:
                self.run = args["run"]
            else:
                logger.warning(f"Ignoring run argument '{args['run']}' - using configured run '{self.run}'")

        # Auto-generate run name if not provided
        if self.run is None:
            self.run = auto_run_name(prefix="sb3-eval")

        # Auto-configure WandB if unconfigured
        if self.wandb == WandbConfig.Unconfigured():
            self.wandb = auto_wandb_config(run=self.run, group=self.group, tags=self.tags)
        elif self.group is not None:
            self.wandb.group = self.group

        # Add informative tags
        if "hpo-lab" not in self.wandb.tags:
            self.wandb.tags.append("hpo-lab")
        if self.env_id not in self.wandb.tags:
            self.wandb.tags.append(self.env_id)
        if self.algorithm not in self.wandb.tags:
            self.wandb.tags.append(self.algorithm)
        if "eval" not in self.wandb.tags:
            self.wandb.tags.append("eval")

        # Ensure an evaluation-appropriate job type
        if not self.wandb.job_type:
            self.wandb.job_type = "sb3-eval"

        temp_model_path: Path | None = None

        try:
            if self.policy_uri is not None:
                if self.model_path:
                    logger.warning(
                        "Both policy_uri and model_path provided; policy_uri will override model_path."
                    )
                self.model_path, temp_model_path = self._prepare_model_from_policy_uri(self.policy_uri)

            eval_config = {
                "env_id": self.env_id,
                "algorithm": self.algorithm,
                "n_eval_episodes": self.n_eval_episodes,
                "deterministic": self.deterministic,
                "success_threshold": self.success_threshold,
                "model_path": self.model_path,
                "policy_uri": self.policy_uri,
            }

            # Initialize WandB context if enabled
            wandb_manager = (
                WandbContext(
                    self.wandb,
                    run_config=eval_config,
                    run_config_name="sb3_eval_config",
                )
                if self.wandb.enabled
                else contextlib.nullcontext(None)
            )

            # Run evaluation
            with wandb_manager as wandb_run:
                metrics = self._evaluate()

                # Log results
                self._log_results(metrics, wandb_run)

                if wandb_run is not None and hasattr(wandb_run, "url"):
                    logger.info(f"WandB run: {wandb_run.url}")

            return 0  # Success

        except Exception as e:
            logger.error(f"Evaluation failed with exception: {e}")
            import traceback
            traceback.print_exc()
            return 1  # Failure
        finally:
            if temp_model_path is not None and temp_model_path.exists():
                try:
                    temp_dir = temp_model_path.parent
                    temp_model_path.unlink(missing_ok=True)
                    # Remove directory if empty
                    if temp_dir.exists():
                        for child in temp_dir.iterdir():
                            break
                        else:
                            temp_dir.rmdir()
                except Exception as cleanup_error:
                    logger.warning(f"Failed to clean up temporary model file: {cleanup_error}")

    def _evaluate(self) -> dict[str, float]:
        """Evaluate the model.

        Returns:
            Dictionary of evaluation metrics
        """
        # Create environment
        render_mode = "human" if self.render else None
        env = make_vec_env(self.env_id, n_envs=1, seed=0, vec_env_kwargs={"render_mode": render_mode})

        if self.model_path:
            # Load trained model
            logger.info(f"Loading model from {self.model_path}")

            # Select the right algorithm class
            algorithm_class = {
                "PPO": PPO,
                "A2C": A2C,
                "SAC": SAC,
            }.get(self.algorithm.upper(), PPO)

            try:
                model = algorithm_class.load(self.model_path, env=env)
            except Exception as e:
                logger.error(f"Failed to load model: {e}")
                logger.info("Creating a random policy for testing")
                model = algorithm_class("MlpPolicy", env, verbose=0)
        else:
            # Use a random policy for testing
            logger.info("No model path provided, using random policy")
            algorithm_class = {
                "PPO": PPO,
                "A2C": A2C,
                "SAC": SAC,
            }.get(self.algorithm.upper(), PPO)
            model = algorithm_class("MlpPolicy", env, verbose=0)

        # Evaluate
        logger.info(f"Evaluating on {self.env_id} for {self.n_eval_episodes} episodes...")

        mean_reward, std_reward = evaluate_policy(
            model,
            env,
            n_eval_episodes=self.n_eval_episodes,
            deterministic=self.deterministic,
            render=self.render,
            return_episode_rewards=False,
        )

        # Calculate success rate if threshold is provided
        success_rate = 0.0
        if self.success_threshold is not None:
            # For this we need episode rewards
            episode_rewards, _ = evaluate_policy(
                model,
                env,
                n_eval_episodes=self.n_eval_episodes,
                deterministic=self.deterministic,
                render=False,  # Don't render second evaluation
                return_episode_rewards=True,
            )
            success_rate = sum(r >= self.success_threshold for r in episode_rewards) / len(episode_rewards)
        else:
            # Default success thresholds for known environments
            default_thresholds = {
                "LunarLander-v3": 200,
                "LunarLander-v2": 200,
                "CartPole-v1": 195,
                "CartPole-v0": 195,
                "Acrobot-v1": -100,
                "MountainCar-v0": -110,
                "Pendulum-v1": -150,
            }

            if self.env_id in default_thresholds:
                threshold = default_thresholds[self.env_id]
                episode_rewards, _ = evaluate_policy(
                    model,
                    env,
                    n_eval_episodes=self.n_eval_episodes,
                    deterministic=self.deterministic,
                    render=False,
                    return_episode_rewards=True,
                )
                success_rate = sum(r >= threshold for r in episode_rewards) / len(episode_rewards)

        # Clean up
        env.close()

        return {
            "mean_reward": mean_reward,
            "std_reward": std_reward,
            "n_episodes": self.n_eval_episodes,
            "success_rate": success_rate,
            "environment": self.env_id,
            "deterministic": self.deterministic,
        }

    def _log_results(self, metrics: dict[str, float], wandb_run: Any | None = None):
        """Log the evaluation results.

        Args:
            metrics: Evaluation metrics dictionary
            wandb_run: Active WandB run, if any
        """
        logger.info("=" * 50)
        logger.info(f"Evaluation Results for {metrics['environment']}")
        logger.info("=" * 50)
        logger.info(f"Episodes evaluated: {metrics['n_episodes']}")
        logger.info(f"Mean reward: {metrics['mean_reward']:.2f} ± {metrics['std_reward']:.2f}")

        if metrics['success_rate'] > 0 or self.success_threshold is not None:
            logger.info(f"Success rate: {metrics['success_rate']:.1%}")

        logger.info(f"Deterministic: {metrics['deterministic']}")

        if self.model_path:
            logger.info(f"Model: {self.model_path}")
        else:
            logger.info("Model: Random policy (no model provided)")

        logger.info("=" * 50)

        if wandb_run is not None:
            import wandb

            log_payload = {
                "eval/mean_reward": metrics["mean_reward"],
                "eval/std_reward": metrics["std_reward"],
                "eval/success_rate": metrics["success_rate"],
                "eval/n_episodes": metrics["n_episodes"],
                "eval/deterministic": metrics["deterministic"],
            }

            wandb.log(log_payload)
            wandb_run.summary.update(log_payload)

    _ARTIFACT_PRIORITY: dict[str, int] = {
        ".zip": 4,
        ".mpt": 3,
        ".pt": 3,
        ".safetensors": 2,
        ".ckpt": 1,
    }

    def _prepare_model_from_policy_uri(self, policy_uri: str) -> tuple[str, Path | None]:
        """Resolve a policy URI into a local model path."""
        parsed = ParsedURI.parse(policy_uri)

        if parsed.scheme in ("", "file"):
            local_path = self._resolve_local_model_path(parsed)
            return str(local_path), None

        if parsed.scheme == "s3":
            resolved_uri = self._resolve_s3_model_uri(parsed)
            temp_dir = Path(tempfile.mkdtemp(prefix="sb3-eval-"))
            with local_copy(resolved_uri) as temp_path:
                source_path = Path(temp_path)
                dest_path = temp_dir / source_path.name
                shutil.copy2(source_path, dest_path)
            return str(dest_path), dest_path

        raise ValueError(f"Unsupported policy URI scheme '{parsed.scheme}' for policy_uri='{policy_uri}'")

    @staticmethod
    def _normalize_policy_uri(policy_uri: str) -> str:
        """Deprecated."""
        logger.warning("EvaluateSB3Tool._normalize_policy_uri is deprecated and returns the input policy URI.")
        return policy_uri

    @classmethod
    def _artifact_priority(cls, name: str) -> int:
        return cls._ARTIFACT_PRIORITY.get(Path(name).suffix.lower(), 0)

    def _resolve_local_model_path(self, parsed: ParsedURI) -> Path:
        local_path = parsed.require_local_path()

        if local_path.is_file():
            if self._artifact_priority(local_path.name) == 0:
                logger.warning(
                    "Local model path %s does not have a recognized extension; proceeding anyway.",
                    local_path,
                )
            return local_path

        if not local_path.exists():
            raise FileNotFoundError(f"Model path not found at {local_path}")

        if local_path.is_dir():
            best_candidate = self._select_best_local_artifact(local_path)
            if best_candidate is None:
                raise FileNotFoundError(
                    f"No model artifacts (.zip/.pt/.mpt/.safetensors) found in directory {local_path}"
                )
            logger.info("Resolved local policy directory %s to %s", local_path, best_candidate)
            return best_candidate

        raise ValueError(f"Unsupported local path type for {local_path}")

    def _resolve_s3_model_uri(self, parsed: ParsedURI) -> str:
        bucket, key = parsed.require_s3()
        if not key:
            raise ValueError("S3 policy URIs must include an object key")

        s3_client = boto3.client("s3")

        if not key.endswith("/"):
            try:
                s3_client.head_object(Bucket=bucket, Key=key)
                return parsed.canonical
            except ClientError as err:
                error_code = err.response.get("Error", {}).get("Code", "")
                if error_code not in {"404", "NoSuchKey", "403"}:
                    raise

        prefix = key.rstrip("/")
        if prefix and not prefix.endswith("/"):
            prefix = f"{prefix}/"

        paginator = s3_client.get_paginator("list_objects_v2")
        best_key: str | None = None
        best_score: tuple[int, float] | None = None

        for page in paginator.paginate(Bucket=bucket, Prefix=prefix):
            for obj in page.get("Contents", []):
                candidate_key = obj["Key"]
                priority = self._artifact_priority(candidate_key)
                if priority == 0:
                    continue
                modified_ts = obj["LastModified"].timestamp()
                score = (priority, modified_ts)
                if best_score is None or score > best_score:
                    best_score = score
                    best_key = candidate_key

        if best_key is None:
            raise FileNotFoundError(
                "No compatible model artifacts (.zip/.pt/.mpt/.safetensors) found under "
                f"s3://{bucket}/{prefix or key}"
            )

        resolved_uri = f"s3://{bucket}/{best_key}"
        if resolved_uri != parsed.canonical:
            logger.info("Resolved S3 policy prefix %s to %s", parsed.canonical, resolved_uri)
        return resolved_uri

    def _select_best_local_artifact(self, directory: Path) -> Path | None:
        best_path: Path | None = None
        best_score: tuple[int, float] | None = None

        for entry in directory.iterdir():
            if entry.is_dir():
                continue
            priority = self._artifact_priority(entry.name)
            if priority == 0:
                continue
            score = (priority, entry.stat().st_mtime)
            if best_score is None or score > best_score:
                best_score = score
                best_path = entry

        return best_path
