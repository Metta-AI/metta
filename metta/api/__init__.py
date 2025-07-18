"""
metta.api package - provides a clean API for Metta training components.

This package exports all the main functions from the submodules for backward compatibility.
"""

# Re-export from agent.py
# Import additional functions from metta.rl.functions
from metta.rl.functions import (
    calculate_prioritized_sampling_params as calculate_anneal_beta,  # Alias for backward compatibility
)
from metta.rl.functions import (
    cleanup_old_policies,
    save_policy_with_metadata,
    wrap_agent_distributed,
)
from metta.rl.functions import (
    should_run as should_run_on_interval,  # If needed
)

from .agent import (
    Agent,
    _get_default_agent_config,
    create_or_load_agent,
)

# Re-export from directories.py
from .directories import (
    RunDirectories,
    save_experiment_config,
    setup_device_and_distributed,
    setup_run_directories,
)

# Re-export from environment.py
from .environment import (
    Environment,
    NavigationBucketedCurriculum,
    PreBuiltConfigCurriculum,
    _get_default_env_config,
)

# Re-export from evaluation.py
from .evaluation import (
    create_evaluation_config_suite,
    create_replay_config,
    evaluate_policy_suite,
    generate_replay_simple,
)


# Backward compatibility aliases and placeholders
def setup_distributed_training(base_device: str = "cuda"):
    """Backward compatibility alias for setup_device_and_distributed."""
    return setup_device_and_distributed(base_device)


def initialize_wandb(
    run_name, run_dir, enabled=True, project=None, entity=None, config=None, job_type="train", tags=None, notes=None
):
    """Initialize wandb - placeholder for backward compatibility."""
    import os

    from omegaconf import DictConfig

    from metta.common.wandb.wandb_context import WandbContext

    if enabled:
        wandb_config = {
            "enabled": True,
            "project": project or os.environ.get("WANDB_PROJECT", "metta"),
            "entity": entity or os.environ.get("WANDB_ENTITY", "metta-research"),
            "group": run_name,
            "name": run_name,
            "run_id": run_name,
            "data_dir": run_dir,
            "job_type": job_type,
            "tags": tags or [],
            "notes": notes or "",
        }
    else:
        wandb_config = {"enabled": False}

    global_config = {
        "run": run_name,
        "run_dir": run_dir,
        "cmd": job_type,
        "wandb": wandb_config,
    }

    if config:
        global_config.update(config)

    wandb_ctx = WandbContext(DictConfig(wandb_config), DictConfig(global_config))
    wandb_run = wandb_ctx.__enter__()

    return wandb_run, wandb_ctx


def cleanup_wandb(wandb_ctx):
    """Clean up wandb context."""
    if wandb_ctx is not None:
        wandb_ctx.__exit__(None, None, None)


def cleanup_distributed():
    """Clean up distributed training."""
    import torch

    if torch.distributed.is_initialized():
        torch.distributed.destroy_process_group()


# Placeholder for missing functions - these need to be implemented
class Optimizer:
    """Wrapper class for optimizer - placeholder for backward compatibility."""

    def __init__(self, optimizer_type, policy, learning_rate, betas, eps, weight_decay, max_grad_norm):
        import torch
        from heavyball import ForeachMuon

        opt_cls = torch.optim.Adam if optimizer_type == "adam" else ForeachMuon
        # ForeachMuon expects int for weight_decay, Adam expects float
        if optimizer_type != "adam":
            weight_decay = int(weight_decay)

        self.optimizer = opt_cls(
            policy.parameters(),
            lr=learning_rate,
            betas=betas,
            eps=eps,
            weight_decay=weight_decay,
        )
        self.max_grad_norm = max_grad_norm
        self.param_groups = self.optimizer.param_groups

    def step(self, loss, epoch, accumulate_minibatches):
        """Optimizer step with gradient accumulation."""
        import torch

        self.optimizer.zero_grad()
        loss.backward()

        if (epoch + 1) % accumulate_minibatches == 0:
            torch.nn.utils.clip_grad_norm_(
                [p for group in self.optimizer.param_groups for p in group["params"]], self.max_grad_norm
            )
            self.optimizer.step()

            if torch.cuda.is_available():
                torch.cuda.synchronize()

    def state_dict(self):
        return self.optimizer.state_dict()

    def load_state_dict(self, state_dict):
        self.optimizer.load_state_dict(state_dict)


def load_checkpoint(checkpoint_dir, agent, optimizer, policy_store, device):
    """Load checkpoint - placeholder for backward compatibility."""
    from metta.rl.trainer_checkpoint import TrainerCheckpoint

    # Try to load existing checkpoint
    existing_checkpoint = TrainerCheckpoint.load(checkpoint_dir)

    if existing_checkpoint:
        agent_step = existing_checkpoint.agent_step
        epoch = existing_checkpoint.epoch

        # Load optimizer state if provided
        if optimizer is not None and existing_checkpoint.optimizer_state_dict:
            try:
                if hasattr(optimizer, "optimizer"):
                    optimizer.optimizer.load_state_dict(existing_checkpoint.optimizer_state_dict)
                elif hasattr(optimizer, "load_state_dict"):
                    optimizer.load_state_dict(existing_checkpoint.optimizer_state_dict)
            except ValueError:
                pass  # Ignore optimizer state mismatch

        return agent_step, epoch, existing_checkpoint.policy_path

    return 0, 0, None


def save_checkpoint(
    epoch,
    agent_step,
    agent,
    optimizer,
    policy_store,
    checkpoint_path,
    checkpoint_interval,
    stats=None,
    force_save=False,
):
    """Save checkpoint - placeholder for backward compatibility."""
    import torch

    from metta.rl.trainer_checkpoint import TrainerCheckpoint

    should_save = force_save or (epoch % checkpoint_interval == 0)
    if not should_save:
        return None

    # Only master saves in distributed mode
    if torch.distributed.is_initialized() and torch.distributed.get_rank() != 0:
        return None

    # Save policy
    saved_record = save_policy_with_metadata(
        policy=agent,
        policy_store=policy_store,
        epoch=epoch,
        agent_step=agent_step,
        evals={},
        timer=None,
        initial_policy_record=None,
        run_name="",
        is_master=True,
    )

    if saved_record:
        # Save training state
        optimizer_state_dict = None
        if optimizer is not None:
            if hasattr(optimizer, "optimizer"):
                optimizer_state_dict = optimizer.optimizer.state_dict()
            elif hasattr(optimizer, "state_dict"):
                optimizer_state_dict = optimizer.state_dict()

        checkpoint = TrainerCheckpoint(
            agent_step=agent_step,
            epoch=epoch,
            optimizer_state_dict=optimizer_state_dict,
            policy_path=saved_record.uri if hasattr(saved_record, "uri") else None,
            stopwatch_state=None,
        )
        checkpoint.save(checkpoint_path)

        # Clean up old policies periodically
        if epoch % 10 == 0:
            cleanup_old_policies(checkpoint_path, keep_last_n=5)

        return saved_record

    return None


def ensure_initial_policy(agent, policy_store, checkpoint_path, loaded_policy_path, device):
    """Ensure initial policy exists - placeholder for backward compatibility."""
    import os

    import torch

    # If we loaded a policy, load it into the agent
    if loaded_policy_path and os.path.exists(loaded_policy_path):
        try:
            policy_pr = policy_store.policy_record(loaded_policy_path)
            agent.load_state_dict(policy_pr.policy.state_dict())
        except Exception:
            pass  # Ignore errors

    # Ensure all ranks synchronize
    if torch.distributed.is_initialized():
        torch.distributed.barrier()


# Make sure all expected exports are available
__all__ = [
    # Agent
    "Agent",
    "create_or_load_agent",
    "_get_default_agent_config",
    # Directories
    "RunDirectories",
    "setup_run_directories",
    "save_experiment_config",
    "setup_device_and_distributed",
    "setup_distributed_training",  # Alias
    # Environment
    "Environment",
    "PreBuiltConfigCurriculum",
    "NavigationBucketedCurriculum",
    "_get_default_env_config",
    # Evaluation
    "create_evaluation_config_suite",
    "create_replay_config",
    "evaluate_policy_suite",
    "generate_replay_simple",
    # Utilities
    "initialize_wandb",
    "cleanup_wandb",
    "cleanup_distributed",
    "wrap_agent_distributed",
    "save_policy_with_metadata",
    "calculate_anneal_beta",
    "cleanup_old_policies",
    "should_run_on_interval",
    "Optimizer",
    "load_checkpoint",
    "save_checkpoint",
    "ensure_initial_policy",
]
