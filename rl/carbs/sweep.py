import contextlib
import fcntl
import os
import random
import time
from dataclasses import dataclass
from typing import Generator

import wandb
import yaml
from carbs import (
    CARBS,
    CARBSParams,
)
from omegaconf import DictConfig, OmegaConf
from rl.carbs.spaces import _carbs_params_spaces, wandb_sweep_cfg
from rl.wandb.wandb_context import WandbContext

@dataclass
class CarbsSweepState:
    wandb_sweep_id: str
    carbs: CARBS
    num_suggestions: int = 0
    num_observations: int = 0
    num_failures: int = 0

@contextlib.contextmanager
def CarbsSweep(sweep_dir: str) -> Generator[CarbsSweepState, None, None]:
    with _carbs_lock(sweep_dir):
        try:
            sweep_state = _load_sweep_state(sweep_dir)
            yield sweep_state
        finally:
            _save_sweep_state(sweep_dir, sweep_state)

@contextlib.contextmanager
def _carbs_lock(sweep_dir: str, max_retries: int = 3, retry_delay: int = 10):
    lock_file = os.path.join(sweep_dir, "carbs.lock")
    lockf = None

    for attempt in range(max_retries):
        try:
            os.makedirs(os.path.dirname(lock_file), exist_ok=True)
            lockf = open(lock_file, 'w')
            fcntl.flock(lockf, fcntl.LOCK_EX | fcntl.LOCK_NB)
            break
        except (IOError, OSError) as e:
            if lockf:
                lockf.close()
            if attempt < max_retries - 1:
                delay = random.uniform(retry_delay, 2 * retry_delay)
                print(f"Failed to acquire CARBS lock, retrying in {delay:.2f} seconds...")
                time.sleep(delay)
            else:
                print(f"Failed to acquire CARBS lock after {max_retries} attempts.")
                raise IOError(f"Failed to acquire CARBS lock after {max_retries} attempts: {str(e)}")

    try:
        yield
    finally:
        if lockf:
            fcntl.flock(lockf, fcntl.LOCK_UN)
            lockf.close()
        os.remove(lock_file)


def _load_sweep_state(sweep_dir: str) -> CarbsSweepState:
    with open(os.path.join(sweep_dir, "sweep.yaml"), "r") as f:
        sweep_state = yaml.safe_load(f)
        carbs = CARBS.load_from_string(sweep_state["carbs"])
        carbs._set_seed(int(time.time()))
        sweep_state["carbs"] = carbs
    return CarbsSweepState(**sweep_state)

def _save_sweep_state(sweep_dir: str, sweep_state: CarbsSweepState):
    with open(os.path.join(sweep_dir, "sweep.yaml"), "w") as f:
        sweep_state_dict = sweep_state.__dict__.copy()
        sweep_state_dict["carbs"] = sweep_state.carbs.serialize()
        yaml.dump(sweep_state_dict, f, default_flow_style=False)

def apply_carbs_suggestion(cfg: OmegaConf, suggestion: DictConfig):
    for key, value in suggestion.items():
        if key == "suggestion_uuid":
            continue
        new_cfg_param = cfg
        key_parts = key.split(".")
        for k in key_parts[:-1]:
            new_cfg_param = new_cfg_param[k]
        param_name = key_parts[-1]
        new_cfg_param[param_name] = value

def pow2_suggestion(cfg: OmegaConf, suggestion: DictConfig):
    new_suggestion = {}
    for key, value in suggestion.items():
        if key == "suggestion_uuid":
            continue
        sweep_param = cfg.sweep.parameters
        key_parts = key.split(".")
        for k in key_parts[:-1]:
            sweep_param = sweep_param[k]
        param_name = key_parts[-1]
        if sweep_param[param_name].space == "pow2":
            value = 2**value
        new_suggestion[key] = value
    return new_suggestion

def create_sweep_state_if_needed(cfg):
    with _carbs_lock(cfg.run_dir):
        _create_sweep_state(cfg)

def _create_sweep_state(cfg):
    if os.path.exists(os.path.join(cfg.run_dir, "sweep.yaml")):
        print(f"Sweep already exists at {cfg.run_dir}")
        return

    os.makedirs(cfg.run_dir, exist_ok=True)

    with WandbContext(cfg) as wandb_ctx:
        wandb_sweep_id = wandb.sweep(
                sweep=wandb_sweep_cfg(cfg),
            project=cfg.wandb.project,
                entity=cfg.wandb.entity,
            )
        wandb.save()
        print(f"WanDb Sweep created with ID: {wandb_sweep_id}")

        carbs_spaces = _carbs_params_spaces(cfg)

        carbs = CARBS(
            CARBSParams(
                better_direction_sign=1,
                    resample_frequency=5,
                    num_random_samples=cfg.sweep.num_random_samples,
                    checkpoint_dir=f"{cfg.run_dir}/carbs/",
                    is_wandb_logging_enabled=False,
                    seed=int(time.time()),
                ),
                carbs_spaces
        )
        carbs_state = CarbsSweepState(
            wandb_sweep_id=wandb_sweep_id,
            carbs=carbs,
        )

    _save_sweep_state(cfg.run_dir, carbs_state)
    print(f"Sweep created at {cfg.run_dir}")
