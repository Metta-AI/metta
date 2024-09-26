from omegaconf import OmegaConf

import wandb
from rl.wandb.wandb import init_wandb

import math
import time
import traceback
from math import ceil, floor, log
from rl.pufferlib.evaluate import evaluate
from rl.pufferlib.dashboard.dashboard import Dashboard
from rl.pufferlib.train import PufferTrainer
from .util import carbs_params_spaces, wandb_sweep_cfg
import yaml
import numpy as np
import torch
from carbs import (
    CARBS,
    CARBSParams,
    LinearSpace,
    LogitSpace,
    LogSpace,
    ObservationInParam,
    Param,
)
from omegaconf import DictConfig, OmegaConf

import wandb
from rl.wandb.wandb import init_wandb
from wandb.errors import CommError

import os

global _controller

def run_carb_sweep_rollout():
    global _controller
    _controller.run_rollout()

class CarbsController:
    def __init__(self, cfg: OmegaConf):
        self._cfg = cfg
        init_wandb(cfg)

        self._state_path = f"{self._cfg.data_dir}/{self._cfg.experiment}/sweep.yaml"

        if cfg.sweep.resume and os.path.exists(self._state_path):
            self.load()
        else:
            self._num_suggestions = 0
            self._num_observations = 0
            self._num_failures = 0
            self._wandb_sweep_id = wandb.sweep(
                sweep=wandb_sweep_cfg(cfg),
                project=cfg.wandb.project,
                entity=cfg.wandb.entity,
            )
            carbs_spaces = carbs_params_spaces(cfg)
            self._carbs = CARBS(
                CARBSParams(
                    better_direction_sign=1,
                    resample_frequency=5,
                    num_random_samples=5,
                    checkpoint_dir=f"{self._cfg.data_dir}/{self._cfg.experiment}/carbs/",
                    is_wandb_logging_enabled=False,
                ),
                carbs_spaces
            )
            self._carbs._set_seed(int(time.time()))
            self.save()
        # if self.cfg.sweep.uri:
        #     self._load_sweep(self.cfg.sweep.uri)

    def run_sweep(self):
        global _controller
        _controller = self
        wandb.agent(self._wandb_sweep_id, function=run_carb_sweep_rollout)

    def run_rollout(self):
        start_time = time.time()
        is_failure = False
        try:
            carbs_suggestion = self._generate_carbs_suggestion()
            self._train(carbs_suggestion)
            score = self._evaluate(carbs_suggestion)
        except Exception as e:
            traceback.print_exc()
            is_failure = True
        rollout_time = time.time() - start_time
        self._record_observation(carbs_suggestion, score, is_failure, rollout_time)

    def load(self):
        with open(self._state_path, "r") as f:
            state = yaml.safe_load(f)
        self._wandb_sweep_id = state["wandb_sweep_id"]
        self._carbs = CARBS.load_from_string(state["carbs"])
        self._num_suggestions = state["num_suggestions"]
        self._num_observations = state["num_observations"]
        self._num_failures = state["num_failures"]

    def save(self):
        with open(self._state_path, "w") as f:
            yaml.dump({
                "wandb_sweep_id": self._wandb_sweep_id,
                "num_suggestions": self._num_suggestions,
                "num_observations": self._num_observations,
                "num_failures": self._num_failures,
                "carbs": self._carbs.serialize(),
            }, f)

    def _train(self):
        trainer = PufferTrainer(run_cfg)
        trainer.train()
        model_artifact_name = trainer._upload_model_to_wandb()
        trainer.close()

    def evaluate(self):
        eval_cfg = self._generate_eval_cfg(carbs_suggestion, model_artifact_name)
        stats = evaluate(eval_cfg)

    def _generate_carbs_suggestion(self):
        self._suggestion = self._carbs.suggest()
        self._num_suggestions += 1
        self._train_cfg = self._cfg.copy()
        self._eval_cfg = self._cfg.copy()


    def _record_observation(self, carbs_suggestion, score, is_failure, rollout_time):
        self._carbs.observe(
            ObservationInParam(
                input=carbs_suggestion,
                output=score,
                cost=rollout_time,
                is_failure=is_failure,
        ))
        self.save()

    def _load_sweep(self, uri: str):
        init_wandb(self._cfg.wandb)
        artifact = wandb.use_artifact(uri, type="sweep")
        self.sweep_id = artifact.metadata["sweep_id"]
        wandb.finish(quiet=True)

        run_cfg = self._cfg.copy()
        run_cfg.train.resume = False
        return run_cfg

    def _generate_eval_cfg(self, carbs_suggestion, model_artifact_name):
        eval_cfg = self._cfg.copy()
        eval_cfg.eval.policy_uri = f"wandb://{model_artifact_name}"
        return eval_cfg

    def _create_new_sweep(self):
        sweep_id = wandb.sweep(
            sweep=self._wandb_sweep_cfg(self._cfg),
            project=self._cfg.wandb.project,
            entity=self._cfg.wandb.entity,
        )
        self._save_carbs_state(self._init_carbs(self._cfg), self._cfg.experiment, sweep_id)

    def _save_carbs_state(self, carbs_suggestion, sweep_id):
        artifact = wandb.Artifact(
            experiment,
            type="sweep",
            metadata={
                "sweep_id": sweep_id,
                "num_observations": carbs_controller.observation_count,
                "num_suggestions": carbs_controller.num_suggestions,
            })
        with artifact.new_file("carbs_state") as f:
            f.write(carbs_controller.serialize())
        artifact.save()

    def _init_carbs(self, cfg):
        pass

    def _carbs_params_spaces(self, cfg: OmegaConf):
        param_spaces = []
        params = _fully_qualified_parameters(cfg.sweep.parameters)
        for param_name, param in params.items():
            train_cfg_param = cfg
            if "search_center" not in param:
                for k in param_name.split("."):
                    train_cfg_param = train_cfg_param[k]
                OmegaConf.set_struct(param, False)
                param.search_center = train_cfg_param
                OmegaConf.set_struct(param, True)

            if param.space == "pow2":
                param.min = int(math.log2(param.min))
                param.max = int(math.log2(param.max))
                param.search_center = int(math.log2(param.search_center))

            scale = param.get("scale", 1)
            if param.space == "pow2" or param.get("is_int", False):
                scale = 4

            if param.search_center < param.min or param.search_center > param.max:
                raise ValueError(f"Search center for {param_name}: {param.search_center} is not in range [{param.min}, {param.max}]")

            param_spaces.append(
                Param(
                    name=param_name,
                    space=_carbs_space[param.space](
                        min=param.min,
                        max=param.max,
                        is_integer=param.get("is_int", False) or param.space == "pow2",
                        rounding_factor=param.get("rounding_factor", 1),
                        scale=scale,
                    ),
                    search_center=param.search_center,
                ))
            return param_spaces


    def _carbs_space(self, param):
        return {
            "log": LogSpace,
            "linear": LinearSpace,
            "pow2": LinearSpace,
            "logit": LogitSpace,
        }[param.space]

    def _fully_qualified_parameters(self, nested_dict, prefix=''):
        qualified_params = {}
        if "space" in nested_dict:
            return {prefix: nested_dict}
        for key, value in nested_dict.items():
            new_prefix = f"{prefix}.{key}" if prefix else key
            if isinstance(value, DictConfig):
                qualified_params.update(self._fully_qualified_parameters(value, new_prefix))
        return qualified_params


    def _wandb_distribution(self, param):
        if param.space == "log":
            return "log_uniform_values"
        elif param.space == "linear":
            return "uniform"
        elif param.space == "logit":
            return "uniform"
        elif param.space == "pow2":
            return "int_uniform"
        elif param.space == "linear":
            if param.is_int:
                return "int_uniform"
            else:
                return "uniform"
