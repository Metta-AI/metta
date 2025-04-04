import logging
import os
import time
from collections import defaultdict
from contextlib import nullcontext

import hydra
import numpy as np
import pufferlib
import pufferlib.utils
import torch
import torch.distributed as dist
from omegaconf import OmegaConf

from util.config import config_from_path
import wandb
from agent.metta_agent import DistributedMettaAgent
from agent.policy_store import PolicyStore
from rl.eval.eval_stats_db import EvalStatsDB
from rl.eval.eval_stats_logger import EvalStatsLogger
from rl.pufferlib.profile import Profile
from rl.pufferlib.trace import save_trace_image
from rl.pufferlib.trainer_checkpoint import TrainerCheckpoint
from rl.pufferlib.vecenv import make_vecenv

from rl.pufferlib.algorithm.ppo import PPO
from rl.pufferlib.algorithm.lstm import LSTM
# from rl.pufferlib.algorithm.p3o import P3O
# from rl.pufferlib.algorithm.kickstarter import Kickstarter
# from rl.pufferlib.algorithm.diayn import DIAYN
# from rl.pufferlib.algorithm.e3b import E3B
# from rl.pufferlib.algorithm.l2_reg import L2Reg



from agent.util.distribution_utils import sample_logits
from tensordict import TensorDict
from heavyball import ForeachMuon
import heavyball.utils

torch.set_float32_matmul_precision('high')

# Get rank for logger name
rank = int(os.environ.get("RANK", 0))
local_rank = int(os.environ.get("LOCAL_RANK", 0))
logger = logging.getLogger(f"trainer-{rank}-{local_rank}")
class PufferTrainer:
    def __init__(self,
                 cfg: OmegaConf,
                 wandb_run,
                 policy_store: PolicyStore,
                 **kwargs):

        self.cfg = cfg
        self.trainer_cfg = cfg.trainer
        self._env_cfg = config_from_path(
            self.trainer_cfg.env, self.trainer_cfg.env_overrides)

        self._master = True
        self._world_size = 1
        self.device = cfg.device
        if dist.is_initialized():
            self._master = (int(os.environ["RANK"]) == 0)
            self._world_size = dist.get_world_size()
            logger.info(f"Rank: {os.environ['RANK']}, Local rank: {os.environ['LOCAL_RANK']}, World size: {self._world_size}")
            self.device = f'cuda:{os.environ["LOCAL_RANK"]}'
            logger.info(f"Setting up distributed training on device {self.device}")

        self.profile = Profile()

        self.stats = defaultdict(list)
        self.wandb_run = wandb_run
        self.policy_store = policy_store
        self.eval_stats_logger = EvalStatsLogger(cfg, self._env_cfg)
        self.average_reward = 0.0  # Initialize average reward estimate
        self._policy_fitness = []
        self._effective_rank = []
        self._make_vecenv()

        logger.info("Loading checkpoint")
        os.makedirs(cfg.trainer.checkpoint_dir, exist_ok=True)
        checkpoint = TrainerCheckpoint.load(cfg.run_dir)

        logger.info("Setting up policy")
        if checkpoint.policy_path:
            logger.info(f"Loading policy from checkpoint: {checkpoint.policy_path}")
            policy_record = policy_store.policy(checkpoint.policy_path)
            if hasattr(checkpoint, 'average_reward'):
                self.average_reward = checkpoint.average_reward
        elif cfg.trainer.initial_policy.uri is not None:
            logger.info(f"Loading initial policy: {cfg.trainer.initial_policy.uri}")
            policy_record = policy_store.policy(cfg.trainer.initial_policy)
        else:
            policy_path = os.path.join(cfg.trainer.checkpoint_dir,
                                 policy_store.make_model_name(0))
            for i in range(20):
                if os.path.exists(policy_path):
                    logger.info(f"Loading policy from checkpoint: {policy_path}")
                    policy_record = policy_store.policy(policy_path)
                    break
                elif self._master:
                    logger.info("Creating new policy")
                    policy_record = policy_store.create(self.vecenv.driver_env)
                    break

                logger.info("No policy found. Waiting for 10 seconds before retrying.")
                time.sleep(10)
            assert policy_record is not None, "No policy found"

        if self._master:
            print(policy_record.policy())

        if policy_record.metadata["action_names"] != self.vecenv.driver_env.action_names():
            raise ValueError(
                "Action names do not match between policy and environment: "
                f"{policy_record.metadata['action_names']} != {self.vecenv.driver_env.action_names()}")

        self._initial_pr = policy_record
        self.last_pr = policy_record
        self.policy = policy_record.policy().to(self.device)
        self.policy_record = policy_record
        self.uncompiled_policy = self.policy

        if self.trainer_cfg.compile:
            logger.info("Compiling policy")
            self.policy = torch.compile(self.policy, mode=self.trainer_cfg.compile_mode)

        if dist.is_initialized():
            logger.info(f"Initializing DistributedDataParallel on device {self.device}")
            # Store the original policy for cleanup purposes
            self._original_policy = self.policy
            self.policy = DistributedMettaAgent(self.policy, self.device)

        self._algs = []
        self._algs.append(PPO(
            gamma=self.trainer_cfg.ppo.gamma,
            gae_lambda=self.trainer_cfg.ppo.gae_lambda,
            clip_vloss=self.trainer_cfg.ppo.clip_vloss,
            vf_clip_coef=self.trainer_cfg.ppo.vf_clip_coef,
            vf_coef=self.trainer_cfg.ppo.vf_coef,
            ent_coef=self.trainer_cfg.ppo.ent_coef,
        ))
        self._algs.append(LSTM(
            num_layers=self.policy.core_num_layers,
            total_agents=self.vecenv.num_agents,
            hidden_size=self.policy.hidden_size
        ))

        # if self.trainer_cfg.diayn.enabled:
        #     self._algs.append(DIAYN(self.trainer_cfg.diayn.archive))
        # if self.trainer_cfg.e3b.enabled:
        #     self._algs.append(E3B(self.policy.hidden_size, self.trainer_cfg.e3b.lambda_, self.trainer_cfg.e3b.norm))
        # if self.trainer_cfg.kickstart.enabled:
        #     self._algs.append(Kickstarter(self.cfg, self.policy_store, self.vecenv.single_action_space))

        batch_size = self.trainer_cfg.batch_size
        self.experience = TensorDict({
            "env": TensorDict({
                "obs": torch.zeros(
                    batch_size,
                    *self.vecenv.single_observation_space.shape,
                    dtype=pufferlib.pytorch.numpy_to_torch_dtype_dict[self.vecenv.single_observation_space.dtype],
                    pin_memory=self.device == 'cuda' and self.trainer_cfg.cpu_offload,
                    device="cpu" if self.trainer_cfg.cpu_offload else self.device
                ),
                "rewards": torch.zeros(batch_size),
                "dones": torch.zeros(batch_size),
                "truncateds": torch.zeros(batch_size),
            }, batch_size=batch_size),
            "policy": TensorDict({
                "logprobs": torch.zeros(batch_size),
                "atn": torch.zeros(
                    batch_size,
                    *self.vecenv.single_action_space.shape,
                    dtype=pufferlib.pytorch.numpy_to_torch_dtype_dict[self.vecenv.single_action_space.dtype]
                ),
            }, batch_size=batch_size),
            "ptr": 0,
            "step": 0,
        })

        for alg in self._algs:
            alg.make_experience_buffers(self.experience)

        self.agent_step = checkpoint.agent_step
        self.epoch = checkpoint.epoch

        assert self.trainer_cfg.optimizer.type in ('adam', 'muon')
        opt_cls = torch.optim.Adam if self.trainer_cfg.optimizer.type == 'adam' else ForeachMuon
        self.optimizer = opt_cls(
            self.policy.parameters(),
            lr=self.trainer_cfg.optimizer.learning_rate,
            betas=(self.trainer_cfg.optimizer.beta1, self.trainer_cfg.optimizer.beta2),
            eps=self.trainer_cfg.optimizer.eps
        )

        epochs = self.trainer_cfg.total_timesteps // self.trainer_cfg.batch_size
        assert self.trainer_cfg.scheduler in ('linear', 'cosine')
        if self.trainer_cfg.scheduler == 'linear':
            self.scheduler = torch.optim.lr_scheduler.LinearLR(
                self.optimizer, start_factor=1.0, end_factor=0.0, total_iters=epochs)
        elif self.trainer_cfg.scheduler == 'cosine':
            self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(self.optimizer, T_max=epochs)

        self.scaler = None if self.trainer_cfg.precision == 'float32' else torch.amp.GradScaler()
        self.amp_context = (nullcontext() if self.trainer_cfg.precision == 'float32'
            else torch.amp.autocast(device_type='cuda', dtype=getattr(torch, self.trainer_cfg.precision)))

        if checkpoint.agent_step > 0:
            self.optimizer.load_state_dict(checkpoint.optimizer_state_dict)

        if self.cfg.wandb.track and wandb_run and self._master:
            wandb_run.define_metric("train/agent_step")
            for k in ["0verview", "env", "losses", "performance", "train"]:
                wandb_run.define_metric(f"{k}/*", step_metric="train/agent_step")

        logger.info(f"PufferTrainer initialization complete on device: {self.device}")

    def train(self):
        self.train_start = time.time()
        logger.info("Starting training")

        # it doesn't make sense to evaluate more often than checkpointing since we need a saved policy to evaluate
        if self.trainer_cfg.evaluate_interval != 0 and self.trainer_cfg.evaluate_interval < self.trainer_cfg.checkpoint_interval:
            self.trainer_cfg.evaluate_interval = self.trainer_cfg.checkpoint_interval

        logger.info(f"Training on {self.device}")
        while self.agent_step < self.trainer_cfg.total_timesteps:
            # Collecting experience
            self._evaluate()

            # Training on collected experience
            self._train()

            # Processing stats
            self._process_stats()

            logger.info(f"Epoch {self.epoch} - {self.agent_step} "\
                        f"({100.00*self.agent_step / self.trainer_cfg.total_timesteps:.2f}%)")

            # Checkpointing trainer
            if self.epoch % self.trainer_cfg.checkpoint_interval == 0:
                self._checkpoint_trainer()
            if self.trainer_cfg.evaluate_interval != 0 and self.epoch % self.trainer_cfg.evaluate_interval == 0:
                self._evaluate_policy()
            if self.cfg.agent.effective_rank_interval != 0 and self.epoch % self.cfg.agent.effective_rank_interval == 0:
                self._effective_rank = self.policy.compute_effective_rank()
            if self.epoch % self.trainer_cfg.wandb_checkpoint_interval == 0:
                self._save_policy_to_wandb()
            if self.cfg.agent.l2_init_weight_update_interval != 0 and self.epoch % self.cfg.agent.l2_init_weight_update_interval == 0:
                self._update_l2_init_weight_copy()
            if (self.trainer_cfg.trace_interval != 0 and
                self.epoch % self.trainer_cfg.trace_interval == 0):
                self._save_trace_to_wandb()

            self._on_train_step()

        self.train_time = time.time() - self.train_start
        self._checkpoint_trainer()
        self._save_policy_to_wandb()
        logger.info(f"Training complete. Total time: {self.train_time:.2f} seconds")

    def _evaluate_policy(self):
        if not self._master:
            return

        self.cfg.eval.policy_uri = self.last_pr.uri
        self.cfg.analyzer.policy_uri = self.last_pr.uri

        eval = hydra.utils.instantiate(
            self.cfg.eval,
            self.policy_store,
            self.last_pr,
            self.cfg.get("run_id", self.wandb_run.id),
            _recursive_ = False)
        stats = eval.evaluate()

        try:
            self.eval_stats_logger.log(stats)
        except Exception as e:
            logger.error(f"Error logging stats: {e}")

        eval_stats_db = EvalStatsDB.from_uri(self.cfg.eval.eval_db_uri, self.cfg.run_dir, self.wandb_run)
        analyzer = hydra.utils.instantiate(self.cfg.analyzer, eval_stats_db)
        _, policy_fitness_records = analyzer.analyze()
        self._policy_fitness = policy_fitness_records


    def _update_l2_init_weight_copy(self):
        self.policy.update_l2_init_weight_copy()

    def _on_train_step(self):
        pass

    @pufferlib.utils.profile
    def _evaluate(self):
        infos = defaultdict(list)

        with self.amp_context:
            while self.experience["ptr"].item() < self.experience["env"].shape[0]:
                with self.profile.env:
                    o, r, d, t, info, agent_id, valid_agents = self.vecenv.recv()

                    state = TensorDict({
                        "env": TensorDict({
                            "obs": torch.as_tensor(o),
                            "rewards": torch.as_tensor(r),
                            "terminals": torch.as_tensor(d),
                            "truncateds": torch.as_tensor(t),
                            "valid_agents": torch.as_tensor(valid_agents),
                            "agent_ids": torch.as_tensor(agent_id),
                            "dones": torch.as_tensor(d + t)
                        }, batch_size=o.shape[0])
                    })
                    self.agent_step += valid_agents.sum()

                with self.profile.eval_copy:
                    state = state.to(self.device, non_blocking=True)

                with self.profile.eval_forward, torch.no_grad():
                    for alg in self._algs:
                        alg.on_pre_step(self.experience, state)

                    logits, value = self.policy(state["env"]["obs"], state)
                    action, logprob, _, normalized_logits = sample_logits(logits)

                    state["policy"].update({
                        "logits": logits,
                        "value": value,
                        "action": action,
                        "logprob": logprob,
                    })

                    for alg in self._algs:
                        alg.on_post_step(self.experience, state)

                with self.profile.eval_copy, torch.no_grad():
                    for alg in self._algs:
                        alg.store_experience(self.experience, state)

                with self.profile.eval_copy:
                    actions = action.cpu().numpy()

                with self.profile.eval_misc:
                    for i in info:
                        for k, v in pufferlib.utils.unroll_nested_dict(i):
                            infos[k].append(v)

                with self.profile.env:
                    self.vecenv.send(actions)

        with self.profile.eval_misc:
            for k, v in infos.items():
                if isinstance(v, np.ndarray):
                    v = v.tolist()
                try:
                    iter(v)
                except TypeError:
                    self.stats[k].append(v)
                else:
                    self.stats[k] += v

        # TODO: Better way to enable multiple collects
        self.experience.ptr = 0
        self.experience.step = 0
        return self.stats, infos

    @pufferlib.utils.profile
    def _train(self):
        data = self
        config, profile, experience = data.cfg, data.profile, data.experience
        train_cfg = data.trainer_cfg
        self.losses = {}
        losses = data.losses

        with profile.train_misc:
            idxs = np.lexsort((self.sort_keys[:, 2], self.sort_keys[:, 1]))
            self.b_idxs_obs = torch.as_tensor(idxs.reshape(
                    self.minibatch_rows, self.num_minibatches, self.bptt_horizon
                ).transpose(1,0,-1)).to(self.obs.device).long()
            self.b_idxs = self.b_idxs_obs.to(self.device)
            self.b_idxs_flat = self.b_idxs.reshape(
                self.num_minibatches, self.minibatch_size)
            self.sort_keys[:, 1:] = 0
            for module in self._algs:
                module.prepare_experience(experience)

        # Optimizing the policy and value network
        total_minibatches = experience.num_minibatches * train_cfg.update_epochs
        accumulate_minibatches = max(1, train_cfg.minibatch_size // train_cfg.max_minibatch_size)
        for epoch in range(train_cfg.update_epochs):
            for mb in range(experience.num_minibatches):
                with profile.train_misc:
                    state = TensorDict("action", experience.b_actions[mb])
                    map(lambda m: m.initialize_state(state), self._algs)

                    mb_experience = experience[mb].to(config.device)
                    map(lambda m: m.update_state(mb_experience, state), self._algs)

                    if config.device == 'cuda':
                        torch.cuda.synchronize()

                with data.amp_context:
                    with profile.train_forward:
                        logits, newvalue = data.policy.forward_train(mb_experience["obs"], state)

                        # xcxc
                        lstm_h = state.lstm_h
                        lstm_c = state.lstm_c
                        if lstm_h is not None:
                            lstm_h = lstm_h.detach()
                        if lstm_c is not None:
                            lstm_c = lstm_c.detach()

                        actions, newlogprob, entropy, normalized_logits = sample_logits(logits, action=atn)

                        if config.device == 'cuda':
                            torch.cuda.synchronize()

                    with profile.train_misc:
                        map(lambda m: m.compute_losses(state, mb, losses), self._algs)
                        loss = sum(losses.values())

                with profile.learn:
                    if data.scaler is None:
                        loss.backward()
                    else:
                        data.scaler.scale(loss).backward()

                    if data.scaler is not None:
                        data.scaler.unscale_(data.optimizer)

                    with torch.no_grad():
                        grads = torch.cat([p.grad.flatten() for p in data.policy.parameters()])
                        grad_var = grads.var(0).mean() * train_cfg.minibatch_size
                        data.msg = f'Gradient variance: {grad_var.item():.3f}'

                    if (mb + 1) % accumulate_minibatches == 0:
                        torch.nn.utils.clip_grad_norm_(data.policy.parameters(), train_cfg.max_grad_norm)

                        if data.scaler is None:
                            data.optimizer.step()
                        else:
                            data.scaler.step(data.optimizer)
                            data.scaler.update()

                        data.optimizer.zero_grad()

                        if config.device == 'cuda':
                            torch.cuda.synchronize()

                with profile.train_misc:
                    for k, v in losses.items():
                        losses[k] += v.item() / total_minibatches
                    losses["clipfrac"] += clipfrac.item() / total_minibatches
                    losses["grad_var"] += grad_var.item() / total_minibatches

            if train_cfg.target_kl is not None:
                if approx_kl > train_cfg.target_kl:
                    break

        with profile.train_misc:
            if train_cfg.anneal_lr:
                data.scheduler.step()

            if train_cfg.use_p3o:
                y_pred = experience.values_mean
                y_true = experience.reward_block
            else:
                y_pred = experience.values
                y_true = experience.returns

            var_y = y_true.var()
            explained_var = torch.nan if var_y == 0 else 1 - (y_true - y_pred).var() / var_y
            losses.explained_variance = explained_var.item()
            data.epoch += 1

            done_training = data.agent_step >= train_cfg.total_timesteps
            # TODO: beter way to get episode return update without clogging dashboard
            # TODO: make this appear faster
            logs = None
            profile.update(
                self.agent_step,
                train_cfg.total_timesteps,
                self._timers
            )

        return logs

    def _checkpoint_trainer(self):
        if not self._master:
            return

        pr = self._checkpoint_policy()
        self.checkpoint = TrainerCheckpoint(
            self.agent_step,
            self.epoch,
            self.optimizer.state_dict(),
            pr.local_path(),
            average_reward=self.average_reward  # Save average reward state
        ).save(self.cfg.run_dir)

    def _checkpoint_policy(self):
        if not self._master:
            return

        name = self.policy_store.make_model_name(self.epoch)

        generation = 0
        if self._initial_pr:
            generation = self._initial_pr.metadata.get("generation", 0) + 1

        self.last_pr = self.policy_store.save(
            name,
            os.path.join(self.cfg.trainer.checkpoint_dir, name),
            self.uncompiled_policy,
            metadata={
                "agent_step": self.agent_step,
                "epoch": self.epoch,
                "run": self.cfg.run,
                "action_names": self.vecenv.driver_env.action_names(),
                "generation": generation,
                "initial_uri": self._initial_pr.uri,
                "train_time": time.time() - self.train_start,
            }
        )
        # this is hacky, but otherwise the initial_pr points
        # at the same policy as the last_pr
        return self.last_pr

    def _save_policy_to_wandb(self):
        if not self._master:
            return

        if self.wandb_run is None:
            return

        pr = self._checkpoint_policy()
        self.policy_store.add_to_wandb_run(self.wandb_run.name, pr)

    def _save_trace_to_wandb(self):
        image_path = f"{self.cfg.run_dir}/traces/trace.{self.epoch}.png"
        save_trace_image(self.cfg, self.last_pr, image_path)
        if self._master:
            wandb.log({"traces/actions": wandb.Image(image_path)})

    def _process_stats(self):
        for k in list(self.stats.keys()):
            v = self.stats[k]
            try:
                v = np.mean(v)
                self.stats[k] = v
            except:
                del self.stats[k]

        # Now synchronize and aggregate stats across processes
        sps = self.profile.SPS
        agent_steps = self.agent_step
        epoch = self.epoch
        learning_rate = self.optimizer.param_groups[0]["lr"]
        losses = {k: v for k, v in vars(self.losses).items() if not k.startswith('_')}
        performance = {k: v for k, v in self.profile}

        overview = {'SPS': sps}
        for k, v in self.trainer_cfg.stats.overview.items():
            if k in self.stats:
                overview[v] = self.stats[k]

        environment = {
            f"env_{k.split('/')[0]}/{'/'.join(k.split('/')[1:])}": v
            for k, v in self.stats.items()
        }

        policy_fitness_metrics = {
            f'pfs/{r["eval"]}:{r["metric"]}': r["fitness"]
            for r in self._policy_fitness
        }

        effective_rank_metrics = {
            f'train/effective_rank/{rank["name"]}': rank["effective_rank"]
            for rank in self._effective_rank
        }

        if self.wandb_run and self.cfg.wandb.track and self._master:
            self.wandb_run.log({
                **{f"overview/{k}": v for k, v in overview.items()},
                **{f"losses/{k}": v for k, v in losses.items()},
                **{f"performance/{k}": v for k, v in performance.items()},
                **environment,
                **policy_fitness_metrics,
                **effective_rank_metrics,
                "train/agent_step": agent_steps,
                "train/epoch": epoch,
                "train/learning_rate": learning_rate,
                "train/average_reward": self.average_reward if self.trainer_cfg.average_reward else None,
            })

        self._policy_fitness = []
        self._effective_rank = []
        self.stats.clear()

    def close(self):
        self.vecenv.close()

    def initial_pr_uri(self):
        return self._initial_pr.uri

    def last_pr_uri(self):
        return self.last_pr.uri


    def _make_vecenv(self):
        """Create a vectorized environment."""
        # Create the vectorized environment
        self.target_batch_size = self.trainer_cfg.forward_pass_minibatch_target_size // self._env_cfg.game.num_agents
        if self.target_batch_size < 2: # pufferlib bug requires batch size >= 2
            self.target_batch_size = 2
        self.batch_size = (self.target_batch_size // self.trainer_cfg.num_workers) * self.trainer_cfg.num_workers

        self.vecenv = make_vecenv(
            self._env_cfg,
            self.cfg.vectorization,
            num_envs = self.batch_size * self.trainer_cfg.async_factor,
            batch_size = self.batch_size,
            num_workers=self.trainer_cfg.num_workers,
            zero_copy=self.trainer_cfg.zero_copy)

        if self.cfg.seed is None:
            self.cfg.seed = np.random.randint(0, 1000000)
        self.vecenv.async_reset(self.cfg.seed)


class AbortingTrainer(PufferTrainer):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def _on_train_step(self):
        if self.wandb_run is None:
            return

        if "abort" not in wandb.Api().run(self.wandb_run.path).tags:
            return

        logger.info("Abort tag detected. Stopping the run.")
        self.cfg.trainer.total_timesteps = int(self.agent_step)
        self.wandb_run.config.update({
            "trainer.total_timesteps": self.cfg.trainer.total_timesteps
        }, allow_val_change=True)
