from omegaconf import OmegaConf

import wandb
from rl.wandb.wandb import init_wandb

import time
import traceback
from rl.pufferlib.evaluate import PufferTournament
from rl.pufferlib.train import PufferTrainer
from .util import carbs_params_spaces, wandb_sweep_cfg
import yaml
from copy import deepcopy
from carbs import (
    CARBS,
    CARBSParams,
    ObservationInParam,
)

from util.logging import remap_io, restore_io
import os

global _controller

def run_carb_sweep_rollout():
    global _controller
    _controller.run_rollout()

class CarbsController:
    def __init__(self, cfg: OmegaConf):
        self._cfg = cfg
        self._train_cfg = None
        self._eval_cfg = None
        self._trainer = None
        self._stage = "init"
        self._sweep_dir = f"{self._cfg.data_dir}/{self._cfg.experiment}"
        self._state_path = f"{self._sweep_dir}/sweep.yaml"
        self._last_rollout_result = None

        init_wandb(cfg)
        if cfg.sweep.resume and os.path.exists(self._state_path):
            self._load()
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
                    num_random_samples=len(carbs_spaces),
                    checkpoint_dir=f"{self._cfg.data_dir}/{self._cfg.experiment}/carbs/",
                    is_wandb_logging_enabled=False,
                ),
                carbs_spaces
            )
            self._carbs._set_seed(int(time.time()))
            self._save()
        # if self.cfg.sweep.uri:
        #     self._load_sweep(self.cfg.sweep.uri)

    def run_sweep(self):
        global _controller
        _controller = self
        wandb.finish(quiet=True)
        wandb.agent(self._wandb_sweep_id,
                    entity=self._cfg.wandb.entity,
                    project=self._cfg.wandb.project,
                    function=run_carb_sweep_rollout,
                    count=self._cfg.sweep.rollout_count)

    def run_rollout(self):
        remap_io(self._sweep_dir)
        start_time = time.time()
        train_time = 0
        eval_time = 0
        is_failure = False
        score = 0
        self._generate_carbs_suggestion()
        init_wandb(self._train_cfg)
        wandb.config.update(self._rollout_params, allow_val_change=True)

        print("Running rollout: ", self._train_cfg.experiment)
        try:
            self._stage = "train"
            self._trainer = PufferTrainer(self._train_cfg)
            print("Training: ", self._train_cfg.experiment)
            train_time = time.time()
            self._trainer.train()
            train_time = time.time() - train_time
            model_artifact_name = self._trainer._upload_model_to_wandb()
            print("Evaluating: ", model_artifact_name)

            self._stage = "eval"
            eval_time = time.time()
            tournament = PufferTournament(self._eval_cfg, self._eval_cfg.eval.policy_uri, self._eval_cfg.eval.baseline_uris)
            stats = tournament.evaluate()
            if self._cfg.sweep.metric not in stats[0]:
                score = 0
            else:
                metric = stats[0][self._cfg.sweep.metric]
                if metric["count"] == 0:
                    score = 0
                else:
                    score = metric["sum"] / metric["count"]

            eval_time = time.time() - eval_time
        except Exception:
            traceback.print_exc()
            is_failure = True
        rollout_time = time.time() - start_time
        failure_msg = "FAILED" if is_failure else ""
        print(f"Rollout {failure_msg} {self._train_cfg.experiment}: {score}, {rollout_time}")
        wandb.finish(quiet=True)

        self._record_observation(score, is_failure, rollout_time, train_time, eval_time)
        self._save()

    def _load(self):
        with open(self._state_path, "r") as f:
            state = yaml.safe_load(f)
        self._wandb_sweep_id = state["wandb_sweep_id"]
        self._carbs = CARBS.load_from_string(state["carbs"])
        self._num_suggestions = state["num_suggestions"]
        self._num_observations = state["num_observations"]
        self._num_failures = state["num_failures"]

    def _save(self):
        with open(self._state_path, "w") as f:
            yaml.dump({
                "wandb_sweep_id": self._wandb_sweep_id,
                "num_suggestions": self._num_suggestions,
                "num_observations": self._num_observations,
                "num_failures": self._num_failures,
                "carbs": self._carbs.serialize(),
            }, f)


    def _generate_carbs_suggestion(self):
        self._load()
        self._suggestion = self._carbs.suggest().suggestion
        self._num_suggestions += 1
        self._train_cfg = deepcopy(self._cfg)
        self._train_cfg.experiment += f".t.{self._num_suggestions}"

        self._rollout_params = self._suggestion.copy()
        del self._rollout_params["suggestion_uuid"]
        for key, value in self._rollout_params.items():
            new_cfg_param = self._train_cfg
            sweep_param = self._cfg.sweep.parameters
            key_parts = key.split(".")
            for k in key_parts[:-1]:
                new_cfg_param = new_cfg_param[k]
                sweep_param = sweep_param[k]
            param_name = key_parts[-1]
            if sweep_param[param_name].space == "pow2":
                value = 2**value
                self._rollout_params[key] = value
            new_cfg_param[param_name] = value

        print(f"Sweep Params: {self._rollout_params}")

        self._eval_cfg = deepcopy(self._train_cfg)
        self._eval_cfg.train = None
        self._eval_cfg.experiment += f".e.{self._num_suggestions}"
        self._eval_cfg.eval = deepcopy(self._cfg.sweep.eval)
        self._eval_cfg.wandb.track = False
        self._save()

    def _record_observation(self, score, is_failure, rollout_time, train_time, eval_time):
        self._load()
        self._last_rollout_result = {
            "score": score,
            "is_failure" : is_failure,
            "rollout_time": rollout_time,
            "train_time": train_time,
            "eval_time": eval_time,
        }
        self._carbs.observe(
            ObservationInParam(
                input=self._suggestion,
                output=score,
                cost=rollout_time,
                is_failure=is_failure,
            )
        )
        self._num_observations += 1
        self._num_failures += 1 if is_failure else 0
        self._save()

    # def _load_sweep(self, uri: str):
    #     init_wandb(self._cfg.wandb)
    #     artifact = wandb.use_artifact(uri, type="sweep")
    #     self.sweep_id = artifact.metadata["sweep_id"]

    #     run_cfg = self._cfg.copy()
    #     run_cfg.train.resume = False
    #     return run_cfg

    # def _save_carbs_state(self, carbs_suggestion, sweep_id):
    #     artifact = wandb.Artifact(
    #         experiment,
    #         type="sweep",
    #         metadata={
    #             "sweep_id": sweep_id,
    #             "num_observations": carbs_controller.observation_count,
    #             "num_suggestions": carbs_controller.num_suggestions,
    #         })
    #     with artifact.new_file("carbs_state") as f:
    #         f.write(carbs_controller.serialize())
    #     artifact.save()
