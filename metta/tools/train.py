#!/usr/bin/env -S uv run
import multiprocessing
import os
import sys
from logging import Logger

import hydra
import torch
import torch.distributed as dist
from omegaconf import DictConfig, OmegaConf
from torch.distributed.elastic.multiprocessing.errors import record

from metta.agent.metta_agent import make_policy
from metta.agent.policy_store import PolicyStore
from metta.app_backend.stats_client import StatsClient
from metta.common.util.config import Config
from metta.common.util.heartbeat import record_heartbeat
from metta.common.util.script_decorators import get_metta_logger, metta_script
from metta.common.wandb.wandb_context import WandbContext, WandbRun
from metta.mettagrid.curriculum.util import curriculum_from_config_path
from metta.mettagrid.mettagrid_env import MettaGridEnv
from metta.rl.curriculum.curriculum_client import CurriculumClient
from metta.rl.curriculum.curriculum_server import CurriculumServer
from metta.rl.trainer import MettaTrainer
from metta.rl.trainer_checkpoint import TrainerCheckpoint
from metta.rl.vecenv import make_vecenv
from metta.sim.simulation_config import SimulationSuiteConfig
from tools.sweep_config_utils import (
    load_train_job_config_with_overrides,
    validate_train_job_config,
)


# TODO: populate this more
class TrainJob(Config):
    __init__ = Config.__init__
    evals: SimulationSuiteConfig
    map_preview_uri: str | None = None


def _calculate_default_num_workers(is_serial: bool) -> int:
    if is_serial:
        return 1

    cpu_count = multiprocessing.cpu_count() or 1

    if torch.cuda.is_available() and torch.distributed.is_initialized():
        num_gpus = torch.cuda.device_count()
    else:
        num_gpus = 1

    ideal_workers = (cpu_count // 2) // num_gpus

    # Round down to nearest power of 2
    num_workers = 1
    while num_workers * 2 <= ideal_workers:
        num_workers *= 2

    return max(1, num_workers)


def _load_or_create_policy(cfg, policy_store, checkpoint, logger):
    """Load policy from checkpoint/config or create new one. Master only."""
    trainer_cfg = cfg.trainer
    
    # Try checkpoint first
    if checkpoint and checkpoint.policy_path:
        logger.info(f"Loading policy from checkpoint: {checkpoint.policy_path}")
        return policy_store.policy_record(checkpoint.policy_path)
    
    # Try initial_policy from config
    if trainer_cfg.initial_policy and (initial_uri := trainer_cfg.initial_policy.uri) is not None:
        logger.info(f"Loading initial policy URI: {initial_uri}")
        return policy_store.policy_record(initial_uri)
    
    # Try default checkpoint path
    policy_path = os.path.join(trainer_cfg.checkpoint.checkpoint_dir, policy_store.make_model_name(0))
    if os.path.exists(policy_path):
        logger.info(f"Loading policy from checkpoint: {policy_path}")
        return policy_store.policy_record(policy_path)
    
    # Create new policy
    logger.info("No existing policy found, creating new one")
    
    # Create a temporary vecenv to get the environment
    curriculum = CurriculumClient.create(cfg.trainer)
    task = curriculum.get_task()
    
    # Create minimal vecenv just to get the driver_env
    temp_vecenv = make_vecenv(
        curriculum,
        cfg.vectorization,
        num_envs=1,
        batch_size=1,
        num_workers=1,
        zero_copy=cfg.trainer.zero_copy,
        is_training=True,
    )
    
    metta_grid_env = temp_vecenv.driver_env
    assert isinstance(metta_grid_env, MettaGridEnv)
    
    # Create and save new policy
    name = policy_store.make_model_name(0)
    pr = policy_store.create_empty_policy_record(name)
    pr.policy = make_policy(metta_grid_env, cfg)
    
    # Initialize the policy to the environment
    features = metta_grid_env.get_observation_features()
    if hasattr(pr.policy, "initialize_to_environment"):
        pr.policy.initialize_to_environment(
            features, metta_grid_env.action_names, metta_grid_env.max_action_args, torch.device(cfg.device)
        )
    else:
        pr.policy.activate_actions(metta_grid_env.action_names, metta_grid_env.max_action_args, torch.device(cfg.device))
    
    saved_pr = policy_store.save(pr)
    logger.info(f"Successfully saved initial policy to {saved_pr.uri}")
    
    temp_vecenv.close()
    
    return saved_pr


def _broadcast_policy_via_nccl(policy_record, device, rank, logger):
    """Broadcast policy from rank 0 to all other ranks using NCCL."""
    if not dist.is_initialized():
        return policy_record
    
    if rank == 0:
        # Master: prepare policy for broadcast
        policy = policy_record.policy
        state_dict = policy.state_dict()
        
        # Broadcast the policy class name
        policy_class_name = policy.__class__.__name__
        class_name_tensor = torch.tensor([ord(c) for c in policy_class_name], dtype=torch.int32, device=device)
        class_name_len = torch.tensor([len(policy_class_name)], dtype=torch.int32, device=device)
        
        dist.broadcast(class_name_len, src=0)
        dist.broadcast(class_name_tensor, src=0)
        
        # Broadcast metadata
        metadata = policy_record.metadata
        import json
        metadata_str = json.dumps(metadata)
        metadata_tensor = torch.tensor([ord(c) for c in metadata_str], dtype=torch.int32, device=device)
        metadata_len = torch.tensor([len(metadata_str)], dtype=torch.int32, device=device)
        
        dist.broadcast(metadata_len, src=0)
        dist.broadcast(metadata_tensor, src=0)
        
        # Broadcast each parameter
        param_names = list(state_dict.keys())
        num_params = torch.tensor([len(param_names)], dtype=torch.int32, device=device)
        dist.broadcast(num_params, src=0)
        
        for param_name in param_names:
            # Broadcast parameter name
            name_tensor = torch.tensor([ord(c) for c in param_name], dtype=torch.int32, device=device)
            name_len = torch.tensor([len(param_name)], dtype=torch.int32, device=device)
            dist.broadcast(name_len, src=0)
            dist.broadcast(name_tensor, src=0)
            
            # Broadcast parameter tensor
            param_tensor = state_dict[param_name].to(device)
            shape = torch.tensor(param_tensor.shape, dtype=torch.int64, device=device)
            shape_len = torch.tensor([len(param_tensor.shape)], dtype=torch.int32, device=device)
            
            dist.broadcast(shape_len, src=0)
            dist.broadcast(shape, src=0)
            dist.broadcast(param_tensor, src=0)
        
        logger.info(f"Rank {rank}: Broadcasted policy with {len(param_names)} parameters")
        return policy_record
        
    else:
        # Worker: receive policy from master
        from metta.agent.policy_store import PolicyStore
        from metta.agent.policy_record import PolicyRecord
        
        # Receive policy class name
        class_name_len = torch.tensor([0], dtype=torch.int32, device=device)
        dist.broadcast(class_name_len, src=0)
        
        class_name_tensor = torch.zeros(class_name_len.item(), dtype=torch.int32, device=device)
        dist.broadcast(class_name_tensor, src=0)
        policy_class_name = ''.join([chr(c) for c in class_name_tensor.cpu().tolist()])
        
        # Receive metadata
        metadata_len = torch.tensor([0], dtype=torch.int32, device=device)
        dist.broadcast(metadata_len, src=0)
        
        metadata_tensor = torch.zeros(metadata_len.item(), dtype=torch.int32, device=device)
        dist.broadcast(metadata_tensor, src=0)
        metadata_str = ''.join([chr(c) for c in metadata_tensor.cpu().tolist()])
        import json
        metadata = json.loads(metadata_str)
        
        # Receive parameters
        num_params = torch.tensor([0], dtype=torch.int32, device=device)
        dist.broadcast(num_params, src=0)
        
        state_dict = {}
        for _ in range(num_params.item()):
            # Receive parameter name
            name_len = torch.tensor([0], dtype=torch.int32, device=device)
            dist.broadcast(name_len, src=0)
            
            name_tensor = torch.zeros(name_len.item(), dtype=torch.int32, device=device)
            dist.broadcast(name_tensor, src=0)
            param_name = ''.join([chr(c) for c in name_tensor.cpu().tolist()])
            
            # Receive parameter shape
            shape_len = torch.tensor([0], dtype=torch.int32, device=device)
            dist.broadcast(shape_len, src=0)
            
            shape = torch.zeros(shape_len.item(), dtype=torch.int64, device=device)
            dist.broadcast(shape, src=0)
            
            # Receive parameter tensor
            param_tensor = torch.zeros(shape.cpu().tolist(), device=device)
            dist.broadcast(param_tensor, src=0)
            
            state_dict[param_name] = param_tensor
        
        logger.info(f"Rank {rank}: Received policy with {len(state_dict)} parameters")
        
        # Store state_dict in metadata for the trainer to use
        metadata["_broadcasted_state_dict"] = state_dict
        
        # Create a dummy policy record - the trainer will create the actual policy
        # and load the state dict
        from metta.agent.policy_store import PolicyStore
        policy_store = PolicyStore(None, None)  # Dummy policy store
        
        policy_record = PolicyRecord(
            policy_store=policy_store,
            run_name="broadcast",
            uri="nccl://broadcast",
            metadata=metadata,
        )
        policy_record.policy = None  # Will be created by trainer
        
        return policy_record


def train(cfg: DictConfig, wandb_run: WandbRun | None, logger: Logger, curriculum_server: CurriculumServer | None, policy_record):
    if torch.distributed.is_initialized():
        world_size = torch.distributed.get_world_size()
        if cfg.trainer.scale_batches_by_world_size:
            cfg.trainer.forward_pass_minibatch_target_size = (
                cfg.trainer.forward_pass_minibatch_target_size // world_size
            )
            cfg.trainer.batch_size = cfg.trainer.batch_size // world_size

    policy_store = PolicyStore(cfg, wandb_run)  # type: ignore[reportArgumentType]
    stats_client: StatsClient | None = get_stats_client(cfg, logger)
    if stats_client is not None:
        stats_client.validate_authenticated()

    train_job = TrainJob(cfg.train_job)

    trainer = MettaTrainer(
        cfg,
        wandb_run=wandb_run,
        policy_store=policy_store,
        sim_suite_config=train_job.evals,
        stats_client=stats_client,
        policy_record=policy_record,  # Pass the policy record
    )

    try:
        trainer.train()
    finally:
        trainer.close()


@hydra.main(config_path="../configs", config_name="train_job", version_base=None)
@metta_script
@record
def main(cfg: DictConfig) -> int:
    record_heartbeat()

    logger = get_metta_logger()

    logger.info(
        f"Training {cfg.run} on "
        + f"{os.environ.get('NODE_INDEX', '0')}: "
        + f"{os.environ.get('LOCAL_RANK', '0')} ({cfg.device})"
    )

    cfg = load_train_job_config_with_overrides(cfg)

    # Validation must be done after merging
    # otherwise trainer's default num_workers: null will be override the values
    # set by _calculate_default_num_workers, and the validation will fail
    if not cfg.trainer.num_workers:
        cfg.trainer.num_workers = _calculate_default_num_workers(cfg.vectorization == "serial")

    cfg = validate_train_job_config(cfg)

    if "LOCAL_RANK" in os.environ and cfg.device.startswith("cuda"):
        logger.info(f"Initializing distributed training with {os.environ['LOCAL_RANK']} {cfg.device}")
        local_rank = int(os.environ["LOCAL_RANK"])
        cfg.device = f"{cfg.device}:{local_rank}"
        dist.init_process_group(backend="nccl")

    logger.info(f"Training {cfg.run} on {cfg.device}")

    # Load checkpoint if it exists
    checkpoint = TrainerCheckpoint.load(cfg.run_dir)
    rank = int(os.environ.get("RANK", "0"))
    device = torch.device(cfg.device)
    
    # Handle policy loading and broadcasting
    policy_record = None
    if rank == 0:
        # Master loads or creates the policy
        policy_store = PolicyStore(cfg, None)  # No wandb_run yet
        policy_record = _load_or_create_policy(cfg, policy_store, checkpoint, logger)
    
    # Broadcast policy from master to all workers via NCCL
    if dist.is_initialized():
        policy_record = _broadcast_policy_via_nccl(policy_record, device, rank, logger)
        logger.info(f"Rank {rank}: Policy broadcast complete")

    curriculum_server = None
    if os.environ.get("RANK", "0") == "0":
        logger.info(f"Train job config: {OmegaConf.to_yaml(cfg, resolve=True)}")

        with open(os.path.join(cfg.run_dir, "config.yaml"), "w") as f:
            OmegaConf.save(cfg, f)

        curriculum = curriculum_from_config_path(cfg.trainer.curriculum, cfg.trainer.env_overrides)

        curriculum_server = CurriculumServer(curriculum=curriculum, port=cfg.trainer.curriculum_server.port)
        curriculum_server.start()

        with WandbContext(cfg.wandb, cfg) as wandb_run:
            train(cfg, wandb_run, logger, curriculum_server, policy_record)
    else:
        train(cfg, None, logger, None, policy_record)

    if dist.is_initialized():
        if curriculum_server is not None:
            curriculum_server.stop()
        dist.destroy_process_group()

    return 0


if __name__ == "__main__":
    sys.exit(main())