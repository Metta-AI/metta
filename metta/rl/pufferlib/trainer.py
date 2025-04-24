import logging
import os
import time
from collections import defaultdict

import hydra
import numpy as np
import pufferlib
import pufferlib.utils
import torch
import torch.distributed as dist
import wandb
from fast_gae import fast_gae
from heavyball import ForeachMuon
from omegaconf import OmegaConf

from metta.agent.metta_agent import DistributedMettaAgent
from metta.agent.policy_store import PolicyStore
from metta.agent.util.weights_analysis import WeightsMetricsHelper
from metta.rl.pufferlib.experience import Experience
from metta.rl.pufferlib.kickstarter import Kickstarter
from metta.rl.pufferlib.profile import Profile
from metta.rl.pufferlib.trainer_checkpoint import TrainerCheckpoint
from metta.sim.eval_stats_db import EvalStatsDB
from metta.sim.eval_stats_logger import EvalStatsLogger
from metta.sim.replay_helper import ReplayHelper
from metta.sim.simulation import SimulationSuite
from metta.sim.simulation_config import SimulationConfig, SimulationSuiteConfig
from metta.sim.vecenv import make_vecenv
from metta.util.config import config_from_path

torch.set_float32_matmul_precision("high")

# Get rank for logger name
rank = int(os.environ.get("RANK", 0))
local_rank = int(os.environ.get("LOCAL_RANK", 0))
logger = logging.getLogger(f"trainer-{rank}-{local_rank}")


class PufferTrainer:
    def __init__(
        self, cfg: OmegaConf, wandb_run, policy_store: PolicyStore, sim_suite_config: SimulationSuiteConfig, **kwargs
    ):
        self.cfg = cfg
        self.trainer_cfg = cfg.trainer
        self._env_cfg = config_from_path(self.trainer_cfg.env, self.trainer_cfg.env_overrides)
        self.sim_suite_config = sim_suite_config

        self._master = True
        self._world_size = 1
        self.device = cfg.device
        if dist.is_initialized():
            self._master = int(os.environ["RANK"]) == 0
            self._world_size = dist.get_world_size()
            logger.info(
                f"Rank: {os.environ['RANK']}, Local rank: {os.environ['LOCAL_RANK']}, World size: {self._world_size}"
            )
            self.device = f"cuda:{os.environ['LOCAL_RANK']}"
            logger.info(f"Setting up distributed training on device {self.device}")

        self.profile = Profile()
        self.losses = self._make_losses()
        self.stats = defaultdict(list)
        self.wandb_run = wandb_run
        self.policy_store = policy_store
        self.use_e3b = self.trainer_cfg.use_e3b
        self.eval_stats_logger = EvalStatsLogger(self.sim_suite_config, wandb_run)
        self.average_reward = 0.0  # Initialize average reward estimate
        self._current_eval_score = None
        self._eval_results = []
        self._weights_helper = WeightsMetricsHelper(cfg)
        self._make_vecenv()

        logger.info("Loading checkpoint")
        os.makedirs(cfg.trainer.checkpoint_dir, exist_ok=True)
        checkpoint = TrainerCheckpoint.load(cfg.run_dir)

        logger.info("Setting up policy")
        if checkpoint.policy_path:
            logger.info(f"Loading policy from checkpoint: {checkpoint.policy_path}")
            policy_record = policy_store.policy(checkpoint.policy_path)
            if hasattr(checkpoint, "average_reward"):
                self.average_reward = checkpoint.average_reward
        elif cfg.trainer.initial_policy.uri is not None:
            logger.info(f"Loading initial policy: {cfg.trainer.initial_policy.uri}")
            policy_record = policy_store.policy(cfg.trainer.initial_policy)
        else:
            policy_path = os.path.join(cfg.trainer.checkpoint_dir, policy_store.make_model_name(0))
            for _i in range(20):
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

        action_names = self.vecenv.driver_env.action_names()
        if policy_record.metadata["action_names"] != action_names:
            raise ValueError(
                "Action names do not match between policy and environment: "
                f"{policy_record.metadata['action_names']} != {action_names}"
            )

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

        self._make_experience_buffer()

        self.agent_step = checkpoint.agent_step
        self.epoch = checkpoint.epoch

        assert self.trainer_cfg.optimizer.type in ("adam", "muon")
        opt_cls = torch.optim.Adam if self.trainer_cfg.optimizer.type == "adam" else ForeachMuon
        self.optimizer = opt_cls(
            self.policy.parameters(),
            lr=self.trainer_cfg.optimizer.learning_rate,
            betas=(self.trainer_cfg.optimizer.beta1, self.trainer_cfg.optimizer.beta2),
            eps=self.trainer_cfg.optimizer.eps,
        )

        self.lr_scheduler = None
        if self.trainer_cfg.lr_scheduler.enabled:
            self.lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
                self.optimizer, T_max=self.trainer_cfg.total_timesteps // self.trainer_cfg.batch_size
            )

        if checkpoint.agent_step > 0:
            self.optimizer.load_state_dict(checkpoint.optimizer_state_dict)

        if self.cfg.wandb.track and wandb_run and self._master:
            wandb_run.define_metric("train/agent_step")
            for k in ["0verview", "env", "losses", "performance", "train"]:
                wandb_run.define_metric(f"{k}/*", step_metric="train/agent_step")

        self.kickstarter = Kickstarter(self.cfg, self.policy_store, self.vecenv.single_action_space)

        replay_sim_config = SimulationConfig(
            env=self.trainer_cfg.env,
            num_envs=1,
            num_episodes=1,
            env_overrides=self.trainer_cfg.env_overrides,
            device=self.device,
            vectorization=self.cfg.vectorization,
        )
        self.replay_helper = ReplayHelper(replay_sim_config, self.last_pr, wandb_run)

        logger.info(f"PufferTrainer initialization complete on device: {self.device}")

    def train(self):
        self.train_start = time.time()
        self.steps_start = self.agent_step

        logger.info("Starting training")

        # it doesn't make sense to evaluate more often than checkpointing since we need a saved policy to evaluate
        if (
            self.trainer_cfg.evaluate_interval != 0
            and self.trainer_cfg.evaluate_interval < self.trainer_cfg.checkpoint_interval
        ):
            raise ValueError("evaluate_interval must be at least as large as checkpoint_interval")

        logger.info(f"Training on {self.device}")
        while self.agent_step < self.trainer_cfg.total_timesteps:
            # Collecting experience
            self._evaluate()

            # Training on collected experience
            self._train()

            # Processing stats
            self._process_stats()

            # log progress
            steps_per_second = (self.agent_step - self.steps_start) / (time.time() - self.train_start)
            remaining_steps = self.trainer_cfg.total_timesteps - self.agent_step
            remaining_time_sec = remaining_steps / steps_per_second

            # Format remaining time in appropriate units
            if remaining_time_sec < 60:
                time_str = f"{remaining_time_sec:.0f} sec"
            elif remaining_time_sec < 3600:
                time_str = f"{remaining_time_sec / 60:.1f} min"
            elif remaining_time_sec < 86400:  # Less than a day
                time_str = f"{remaining_time_sec / 3600:.1f} hours"
            else:
                time_str = f"{remaining_time_sec / 86400:.1f} days"

            logger.info(
                f"Epoch {self.epoch} - {self.agent_step} [{steps_per_second:.0f}/sec]"
                f" ({100.00 * self.agent_step / self.trainer_cfg.total_timesteps:.2f}%)"
                f" - {time_str} remaining"
            )

            # Checkpointing trainer
            if self.epoch % self.trainer_cfg.checkpoint_interval == 0:
                self._checkpoint_trainer()
            if self.trainer_cfg.evaluate_interval != 0 and self.epoch % self.trainer_cfg.evaluate_interval == 0:
                self._evaluate_policy()
            self._weights_helper.on_epoch_end(self.epoch, self.policy)
            if self.epoch % self.trainer_cfg.wandb_checkpoint_interval == 0:
                self._save_policy_to_wandb()
            if (
                self.cfg.agent.l2_init_weight_update_interval != 0
                and self.epoch % self.cfg.agent.l2_init_weight_update_interval == 0
            ):
                self._update_l2_init_weight_copy()
            if self.trainer_cfg.replay_interval != 0 and self.epoch % self.trainer_cfg.replay_interval == 0:
                self._generate_and_upload_replay()

            self._on_train_step()

        self.train_time = time.time() - self.train_start
        self._checkpoint_trainer()
        self._save_policy_to_wandb()
        logger.info(f"Training complete. Total time: {self.train_time:.2f} seconds")

    def _evaluate_policy(self):
        if not self._master:
            return

        self.cfg.analyzer.policy_uri = self.last_pr.uri

        run_id = self.cfg.get("run_id")
        if run_id is None and self.wandb_run is not None:
            run_id = self.wandb_run.id

        logger.info(f"Simulating policy: {self.last_pr.uri} with config: {self.sim_suite_config}")
        sim = SimulationSuite(config=self.sim_suite_config, policy_pr=self.last_pr, policy_store=self.policy_store)
        stats = sim.simulate()
        logger.info("Simulation complete")

        try:
            self.eval_stats_logger.log(stats)
        except Exception as e:
            logger.error(f"Error logging stats: {e}")

        eval_stats_db = EvalStatsDB.from_uri(self.sim_suite_config.eval_db_uri, self.cfg.run_dir, self.wandb_run)
        analyzer = hydra.utils.instantiate(self.cfg.analyzer, eval_stats_db)
        _, policy_fitness_records = analyzer.analyze()
        self._eval_results = policy_fitness_records
        self._current_eval_score = np.sum(
            [r["candidate_mean"] for r in self._eval_results if r["metric"] == "episode_reward"]
        )

    def _update_l2_init_weight_copy(self):
        self.policy.update_l2_init_weight_copy()

    def _on_train_step(self):
        pass

    @pufferlib.utils.profile
    def _evaluate(self):
        experience, profile = self.experience, self.profile

        with profile.eval_misc:
            policy = self.policy
            infos = defaultdict(list)
            lstm_h, lstm_c = experience.lstm_h, experience.lstm_c
            e3b_inv = experience.e3b_inv

        while not experience.full:
            with profile.env:
                o, r, d, t, info, env_id, mask = self.vecenv.recv()

                # Zero-copy indexing for contiguous env_id

                # This was originally self.config.env_batch_size == 1, but you have scaling
                # configured differently in metta. You want the whole forward pass batch to come
                # from one core to reduce indexing overhead.
                # contiguous_env_ids = self.vecenv.agents_per_batch == self.vecenv.driver_env.agents_per_env[0]
                contiguous_env_ids = self.trainer_cfg.async_factor == self.trainer_cfg.num_workers
                contiguous_env_ids = False
                if contiguous_env_ids:
                    gpu_env_id = cpu_env_id = slice(env_id[0], env_id[-1] + 1)
                else:
                    if self.trainer_cfg.require_contiguous_env_ids:
                        raise ValueError(
                            "Env ids are not contiguous. "
                            f"{self.trainer_cfg.async_factor} != {self.trainer_cfg.num_workers}"
                        )
                    cpu_env_id = env_id
                    gpu_env_id = torch.as_tensor(env_id).to(self.device, non_blocking=True)

            with profile.eval_misc:
                num_steps = sum(mask)
                self.agent_step += num_steps * self._world_size

                o = torch.as_tensor(o)
                o_device = o.to(self.device, non_blocking=True)
                r = torch.as_tensor(r)
                d = torch.as_tensor(d)

            with profile.eval_forward, torch.no_grad():
                # TODO: In place-update should be faster. Leaking 7% speed max
                # Also should be using a cuda tensor to index
                e3b = e3b_inv[gpu_env_id] if self.use_e3b else None

                h = lstm_h[:, gpu_env_id]
                c = lstm_c[:, gpu_env_id]
                actions, logprob, _, value, (h, c), next_e3b, intrinsic_reward, _ = policy(o_device, (h, c), e3b=e3b)
                lstm_h[:, gpu_env_id] = h
                lstm_c[:, gpu_env_id] = c
                if self.use_e3b:
                    e3b_inv[env_id] = next_e3b
                    r += intrinsic_reward.cpu()

                if self.device == "cuda":
                    torch.cuda.synchronize()

            with profile.eval_misc:
                value = value.flatten()
                actions = actions.cpu().numpy()
                mask = torch.as_tensor(mask)  # * policy.mask)
                o = o if self.trainer_cfg.cpu_offload else o_device
                self.experience.store(o, value, actions, logprob, r, d, cpu_env_id, mask)

                for i in info:
                    for k, v in pufferlib.utils.unroll_nested_dict(i):
                        infos[k].append(v)

            with profile.env:
                self.vecenv.send(actions)

        with profile.eval_misc:
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
        experience.ptr = 0
        experience.step = 0
        return self.stats, infos

    @pufferlib.utils.profile
    def _train(self):
        experience, profile = self.experience, self.profile
        self.losses = self._make_losses()

        with profile.train_misc:
            idxs = experience.sort_training_data()
            dones_np = experience.dones_np[idxs]
            values_np = experience.values_np[idxs]
            rewards_np = experience.rewards_np[idxs]

            # Update average reward estimate
            if self.trainer_cfg.average_reward:
                # Update average reward estimate using EMA with configured alpha
                alpha = self.trainer_cfg.average_reward_alpha
                self.average_reward = (1 - alpha) * self.average_reward + alpha * np.mean(rewards_np)
                # Adjust rewards by subtracting average reward for advantage computation
                rewards_np_adjusted = rewards_np - self.average_reward
                # Set gamma to 1.0 for average reward case
                effective_gamma = 1.0
                # Compute advantages using adjusted rewards
                advantages_np = fast_gae.compute_gae(
                    dones_np, values_np, rewards_np_adjusted, effective_gamma, self.trainer_cfg.gae_lambda
                )
                # For average reward case, returns are computed differently:
                # R(s) = Σ(r_t - ρ) represents the bias function
                experience.returns_np = advantages_np + values_np
            else:
                effective_gamma = self.trainer_cfg.gamma
                # Standard GAE computation for discounted case
                advantages_np = fast_gae.compute_gae(
                    dones_np, values_np, rewards_np, effective_gamma, self.trainer_cfg.gae_lambda
                )
                experience.returns_np = advantages_np + values_np

            experience.flatten_batch(advantages_np)

        # Optimizing the policy and value network
        total_minibatches = experience.num_minibatches * self.trainer_cfg.update_epochs
        for _epoch in range(self.trainer_cfg.update_epochs):
            lstm_state = None
            teacher_lstm_state = None
            for mb in range(experience.num_minibatches):
                with profile.train_misc:
                    obs = experience.b_obs[mb]
                    obs = obs.to(self.device, non_blocking=True)
                    atn = experience.b_actions[mb]
                    log_probs = experience.b_logprobs[mb]
                    val = experience.b_values[mb]
                    adv = experience.b_advantages[mb]
                    ret = experience.b_returns[mb]

                with profile.train_forward:
                    _, newlogprob, entropy, newvalue, lstm_state, _, _, new_normalized_logits = self.policy(
                        obs, state=lstm_state, action=atn
                    )
                    lstm_state = (lstm_state[0].detach(), lstm_state[1].detach())

                    if self.device == "cuda":
                        torch.cuda.synchronize()

                with profile.train_misc:
                    logratio = newlogprob - log_probs.reshape(-1)
                    ratio = logratio.exp()

                    with torch.no_grad():
                        # calculate approx_kl http://joschu.net/blog/kl-approx.html
                        old_approx_kl = (-logratio).mean()
                        approx_kl = ((ratio - 1) - logratio).mean()
                        clipfrac = ((ratio - 1.0).abs() > self.trainer_cfg.clip_coef).float().mean()

                    adv = adv.reshape(-1)
                    if self.trainer_cfg.norm_adv:
                        adv = (adv - adv.mean()) / (adv.std() + 1e-8)

                    # Policy loss
                    pg_loss1 = -adv * ratio
                    pg_loss2 = -adv * torch.clamp(ratio, 1 - self.trainer_cfg.clip_coef, 1 + self.trainer_cfg.clip_coef)
                    pg_loss = torch.max(pg_loss1, pg_loss2).mean()

                    # Value loss
                    newvalue = newvalue.view(-1)
                    if self.trainer_cfg.clip_vloss:
                        v_loss_unclipped = (newvalue - ret) ** 2
                        v_clipped = val + torch.clamp(
                            newvalue - val,
                            -self.trainer_cfg.vf_clip_coef,
                            self.trainer_cfg.vf_clip_coef,
                        )
                        v_loss_clipped = (v_clipped - ret) ** 2
                        v_loss_max = torch.max(v_loss_unclipped, v_loss_clipped)
                        v_loss = 0.5 * v_loss_max.mean()
                    else:
                        v_loss = 0.5 * ((newvalue - ret) ** 2).mean()

                    entropy_loss = entropy.mean()

                    ks_action_loss, ks_value_loss, teacher_lstm_state = self.kickstarter.loss(
                        self.agent_step, new_normalized_logits, newvalue, obs, teacher_lstm_state
                    )

                    l2_reg_loss = torch.tensor(0.0, device=self.device)
                    if self.trainer_cfg.l2_reg_loss_coef > 0:
                        l2_reg_loss = self.trainer_cfg.l2_reg_loss_coef * self.policy.l2_reg_loss().to(self.device)

                    l2_init_loss = torch.tensor(0.0, device=self.device)
                    if self.trainer_cfg.l2_init_loss_coef > 0:
                        l2_init_loss = self.trainer_cfg.l2_init_loss_coef * self.policy.l2_init_loss().to(self.device)

                    loss = (
                        pg_loss
                        - self.trainer_cfg.ent_coef * entropy_loss
                        + v_loss * self.trainer_cfg.vf_coef
                        + l2_reg_loss
                        + l2_init_loss
                        + ks_action_loss
                        + ks_value_loss
                    )

                with profile.learn:
                    self.optimizer.zero_grad()
                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(self.policy.parameters(), self.trainer_cfg.max_grad_norm)
                    self.optimizer.step()

                    if self.cfg.agent.clip_range > 0:
                        self.policy.clip_weights()

                    if self.device == "cuda":
                        torch.cuda.synchronize()

                with profile.train_misc:
                    self.losses.policy_loss += pg_loss.item() / total_minibatches
                    self.losses.value_loss += v_loss.item() / total_minibatches
                    self.losses.entropy += entropy_loss.item() / total_minibatches
                    self.losses.old_approx_kl += old_approx_kl.item() / total_minibatches
                    self.losses.approx_kl += approx_kl.item() / total_minibatches
                    self.losses.clipfrac += clipfrac.item() / total_minibatches
                    self.losses.l2_reg_loss += l2_reg_loss.item() / total_minibatches
                    self.losses.l2_init_loss += l2_init_loss.item() / total_minibatches
                    self.losses.ks_action_loss += ks_action_loss.item() / total_minibatches
                    self.losses.ks_value_loss += ks_value_loss.item() / total_minibatches

            if self.trainer_cfg.target_kl is not None:
                if approx_kl > self.trainer_cfg.target_kl:
                    break

        with profile.train_misc:
            if self.lr_scheduler is not None:
                self.lr_scheduler.step()

            y_pred = experience.values_np
            y_true = experience.returns_np
            var_y = np.var(y_true)
            explained_var = np.nan if var_y == 0 else 1 - np.var(y_true - y_pred) / var_y
            self.losses.explained_variance = explained_var
            self.epoch += 1
            profile.update(self.agent_step, self.trainer_cfg.total_timesteps, self._timers)

    def _checkpoint_trainer(self):
        if not self._master:
            return

        pr = self._checkpoint_policy()
        self.checkpoint = TrainerCheckpoint(
            self.agent_step,
            self.epoch,
            self.optimizer.state_dict(),
            pr.local_path(),
            average_reward=self.average_reward,  # Save average reward state
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
                "score": self._current_eval_score,
            },
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

    def _generate_and_upload_replay(self):
        if self._master:
            logger.info("Generating and saving a replay to wandb and S3.")
            self.replay_helper.generate_and_upload_replay(
                self.epoch,
                self.cfg.run_dir,
                self.cfg.run,
                dry_run=self.trainer_cfg.get("replay_dry_run", False),
            )

    def _process_stats(self):
        for k in list(self.stats.keys()):
            v = self.stats[k]
            try:
                v = np.mean(v)
                self.stats[k] = v
            except (TypeError, ValueError):
                del self.stats[k]

        # Now synchronize and aggregate stats across processes
        sps = self.profile.SPS
        agent_steps = self.agent_step
        epoch = self.epoch
        learning_rate = self.optimizer.param_groups[0]["lr"]
        losses = {k: v for k, v in vars(self.losses).items() if not k.startswith("_")}
        performance = {k: v for k, v in self.profile}

        overview = {"SPS": sps}
        for k, v in self.trainer_cfg.stats.overview.items():
            if k in self.stats:
                overview[v] = self.stats[k]

        navigation_score = np.mean([r["candidate_mean"] for r in self._eval_results if "navigation" in r["eval"]])
        object_use_score = np.mean([r["candidate_mean"] for r in self._eval_results if "object_use" in r["eval"]])
        against_npc_score = np.mean([r["candidate_mean"] for r in self._eval_results if "npc" in r["eval"]])

        if not np.isnan(navigation_score):
            overview["navigation_evals"] = navigation_score
        if not np.isnan(object_use_score):
            overview["object_use_evals"] = object_use_score
        if not np.isnan(against_npc_score):
            overview["npc_evals"] = against_npc_score

        environment = {f"env_{k.split('/')[0]}/{'/'.join(k.split('/')[1:])}": v for k, v in self.stats.items()}

        policy_fitness_metrics = {
            f"pfs/{r['eval'].split('/')[-1]}:{r['metric']}": r["fitness"] for r in self._eval_results
        }

        navigation_eval_metrics = {
            f"navigation_evals/{r['eval'].split('/')[-1]}:{r['metric']}": r["candidate_mean"]
            for r in self._eval_results
            if "navigation" in r["eval"]
        }

        object_use_eval_metrics = {
            f"object_use_evals/{r['eval'].split('/')[-1]}:{r['metric']}": r["candidate_mean"]
            for r in self._eval_results
            if "object_use" in r["eval"]
        }

        against_npc_eval_metrics = {
            f"npc_evals/{r['eval'].split('/')[-1]}:{r['metric']}": r["candidate_mean"]
            for r in self._eval_results
            if "npc" in r["eval"]
        }

        if self.wandb_run and self.cfg.wandb.track and self._master:
            self.wandb_run.log(
                {
                    **{f"overview/{k}": v for k, v in overview.items()},
                    **{f"losses/{k}": v for k, v in losses.items()},
                    **{f"performance/{k}": v for k, v in performance.items()},
                    **environment,
                    **policy_fitness_metrics,
                    **self._weights_helper.stats(),
                    **navigation_eval_metrics,
                    **object_use_eval_metrics,
                    **against_npc_eval_metrics,
                    "train/agent_step": agent_steps,
                    "train/epoch": epoch,
                    "train/learning_rate": learning_rate,
                    "train/average_reward": self.average_reward if self.trainer_cfg.average_reward else None,
                }
            )

        self._eval_results = []
        self._weights_helper.reset()
        self.stats.clear()

    def close(self):
        self.vecenv.close()

    def initial_pr_uri(self):
        return self._initial_pr.uri

    def last_pr_uri(self):
        return self.last_pr.uri

    def _make_experience_buffer(self):
        obs_shape = self.vecenv.single_observation_space.shape
        obs_dtype = self.vecenv.single_observation_space.dtype
        atn_shape = self.vecenv.single_action_space.shape
        atn_dtype = self.vecenv.single_action_space.dtype
        total_agents = self.vecenv.num_agents

        self.experience = Experience(
            self.trainer_cfg.batch_size,
            self.trainer_cfg.bptt_horizon,
            self.trainer_cfg.minibatch_size,
            self.policy.hidden_size,
            obs_shape,
            obs_dtype,
            atn_shape,
            atn_dtype,
            self.trainer_cfg.cpu_offload,
            self.device,
            self.policy.lstm,
            total_agents,
        )

    def _make_losses(self):
        return pufferlib.namespace(
            policy_loss=0,
            value_loss=0,
            entropy=0,
            old_approx_kl=0,
            approx_kl=0,
            clipfrac=0,
            explained_variance=0,
            l2_reg_loss=0,
            l2_init_loss=0,
            ks_action_loss=0,
            ks_value_loss=0,
        )

    def _make_vecenv(self):
        """Create a vectorized environment."""
        # Create the vectorized environment
        self.target_batch_size = self.trainer_cfg.forward_pass_minibatch_target_size // self._env_cfg.game.num_agents
        if self.target_batch_size < 2:  # pufferlib bug requires batch size >= 2
            self.target_batch_size = 2
        self.batch_size = (self.target_batch_size // self.trainer_cfg.num_workers) * self.trainer_cfg.num_workers

        self.vecenv = make_vecenv(
            self._env_cfg,
            self.cfg.vectorization,
            num_envs=self.batch_size * self.trainer_cfg.async_factor,
            batch_size=self.batch_size,
            num_workers=self.trainer_cfg.num_workers,
            zero_copy=self.trainer_cfg.zero_copy,
        )

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
        self.wandb_run.config.update(
            {"trainer.total_timesteps": self.cfg.trainer.total_timesteps}, allow_val_change=True
        )
