import logging
from typing import Any

import torch
from pydantic import ConfigDict
from tensordict import NonTensorData, TensorDict

from metta.agent.policy import Policy
from metta.rl.advantage import compute_advantage, compute_delta_lambda
from metta.rl.nodes.base import NodeBase
from metta.rl.nodes.registry import NodeSpec
from metta.rl.training import ComponentContext, Experience, TrainingEnvironment
from metta.rl.training.graph import Node, TrainingGraph
from metta.rl.utils import add_dummy_loss_for_unused_params, ensure_sequence_metadata, forward_policy_for_training
from mettagrid.base_config import Config

logger = logging.getLogger(__name__)


class RolloutResult(Config):
    """Results from a rollout phase."""

    model_config = ConfigDict(arbitrary_types_allowed=True)

    raw_infos: list[dict[str, Any]]
    agent_steps: int
    training_env_id: slice


class CoreTrainingLoop:
    """Handles the core training loop with rollout and training phases."""

    def __init__(
        self,
        policy: Policy,
        experience: Experience,
        nodes: dict[str, NodeBase],
        node_specs: list[NodeSpec],
        optimizer: torch.optim.Optimizer,
        device: torch.device,
        context: ComponentContext,
    ):
        """Initialize core training loop.

        Args:
            policy: The policy to train
            experience: Experience buffer for storing rollouts
        nodes: Dictionary of node instances to use
            optimizer: Optimizer for policy updates
            device: Device to run on
        """
        self.policy = policy
        self.experience = experience
        self.nodes = nodes
        self.node_specs = {spec.key: spec for spec in node_specs}
        self.node_specs_order = list(node_specs)
        self.optimizer = optimizer
        self.device = device
        self.accumulate_minibatches = experience.accumulate_minibatches
        self.context = context
        self.graph = TrainingGraph(_build_nodes(self, nodes, self.node_specs_order))
        self.last_action = torch.zeros(
            experience.total_agents,
            1,
            dtype=torch.int32,
            device=device,
        )
        # Cache environment indices to avoid reallocating per rollout batch
        self._env_index_cache = experience._range_tensor.to(device=device, dtype=torch.long)
        # Get policy spec for experience buffer
        self.policy_spec = policy.get_agent_experience_spec()

    def rollout_phase(
        self,
        env: TrainingEnvironment,
        context: ComponentContext,
    ) -> RolloutResult:
        """Perform rollout phase to collect experience.

        Args:
            env: Vectorized environment to collect from
            context: Shared trainer context providing rollout state

        Returns:
            RolloutResult with collected info
        """
        raw_infos: list[dict[str, Any]] = []
        self.experience.reset_for_rollout()

        # Notify nodes of rollout start
        for name, node in self.nodes.items():
            spec = self.node_specs[name]
            if spec.has_rollout:
                node.on_rollout_start(context)

        # Get buffer for storing experience
        buffer_step = self.experience.buffer[self.experience.row_slot_ids, self.experience.t_in_row - 1]
        buffer_step = buffer_step.select(*self.policy_spec.keys())

        total_steps = 0
        last_env_id: slice | None = None

        while not self.experience.ready_for_training:
            workspace = {
                "env": env,
                "buffer_step": buffer_step,
                "raw_infos": raw_infos,
                "total_steps": total_steps,
            }
            workspace["phase"] = "rollout"
            self.graph.run(context, workspace)
            total_steps = workspace["total_steps"]
            last_env_id = workspace.get("last_env_id", last_env_id)

        return RolloutResult(raw_infos=raw_infos, agent_steps=total_steps, training_env_id=last_env_id)

    def training_phase(
        self,
        context: ComponentContext,
        update_epochs: int,
        max_grad_norm: float = 0.5,
    ) -> tuple[dict[str, float], int]:
        """Perform training phase on collected experience.

        Args:
            context: Shared trainer context providing training state
            update_epochs: Number of epochs to train for
            max_grad_norm: Maximum gradient norm for clipping

        Returns:
            Dictionary of node statistics
        """
        self.experience.reset_importance_sampling_ratios()

        for node in self.nodes.values():
            node.zero_loss_tracker()

        advantage_cfg = context.config.advantage
        ppo_critic = self.nodes.get("ppo_critic")
        use_delta_lambda = (
            ppo_critic is not None
            and getattr(ppo_critic.cfg, "critic_update", None) == "gtd_lambda"
            and ppo_critic.cfg.enabled
            and ppo_critic.node_gate_allows("train", context)
        )
        advantage_method = "delta_lambda" if use_delta_lambda else "vtrace"

        epochs_trained = 0

        for _ in range(update_epochs):
            if "values" in self.experience.buffer.keys():
                values_for_adv = self.experience.buffer["values"]
                if values_for_adv.dim() > 2:
                    values_for_adv = values_for_adv.mean(dim=-1)
                centered_rewards = self.experience.buffer["rewards"] - self.experience.buffer["reward_baseline"]
                advantages_full = compute_advantage(
                    values_for_adv,
                    centered_rewards,
                    self.experience.buffer["dones"],
                    torch.ones_like(values_for_adv),
                    torch.zeros_like(values_for_adv, device=self.device),
                    advantage_cfg.gamma,
                    advantage_cfg.gae_lambda,
                    self.device,
                    advantage_cfg.vtrace_rho_clip,
                    advantage_cfg.vtrace_c_clip,
                )
            else:
                # Value-free setups still need a tensor shaped like the buffer for sampling.
                advantages_full = torch.zeros(
                    self.experience.buffer.batch_size,
                    device=self.device,
                    dtype=torch.float32,
                )
            self.experience.buffer["advantages_full"] = advantages_full

            stop_update_epoch = False
            for mb_idx in range(self.experience.num_minibatches):
                if mb_idx % self.accumulate_minibatches == 0:
                    self.optimizer.zero_grad(set_to_none=True)

                workspace = {
                    "mb_idx": mb_idx,
                    "advantages_full": advantages_full,
                    "stop_update_epoch": False,
                    "advantage_method": advantage_method,
                    "max_grad_norm": max_grad_norm,
                }
                workspace["phase"] = "train"
                self.graph.run(context, workspace)
                stop_update_epoch_mb = bool(workspace["stop_update_epoch"])

                if stop_update_epoch_mb:
                    stop_update_epoch = True
                    break

            epochs_trained += 1
            if stop_update_epoch:
                break

        # Notify nodes of training phase end
        for name, node in self.nodes.items():
            spec = self.node_specs[name]
            if spec.has_train:
                node.on_train_phase_end(context)

        # Collect statistics from all nodes
        graph_stats: dict[str, float] = {}
        for _node_name, node in self.nodes.items():
            graph_stats.update(node.stats())

        return graph_stats, epochs_trained

    def on_epoch_start(self, context: ComponentContext | None = None) -> None:
        """Called at the start of each epoch.

        Args:
            context: Shared trainer context providing epoch state
        """
        for node in self.nodes.values():
            node.on_epoch_start(context)

    def add_last_action_to_td(self, td: TensorDict) -> None:
        env_ids = td["training_env_ids"].squeeze(-1)

        if self.last_action.device != td.device:
            self.last_action = self.last_action.to(device=td.device)

        td["last_actions"] = self.last_action[env_ids].detach()


def _build_nodes(
    core: CoreTrainingLoop,
    nodes: dict[str, NodeBase],
    node_specs: list[NodeSpec],
) -> list[Node]:
    return _build_rollout_nodes(core, nodes, node_specs) + _build_train_nodes(core, nodes, node_specs)


def _build_rollout_nodes(
    core: CoreTrainingLoop,
    nodes: dict[str, NodeBase],
    node_specs: list[NodeSpec],
) -> list[Node]:
    graph_nodes: list[Node] = [
        _rollout_env_wait_node(core),
        _rollout_td_prep_node(core),
    ]

    rollout_node_names: list[str] = []
    base_dep = "rollout.td_prep"
    for spec in node_specs:
        if not spec.has_rollout:
            continue
        node = nodes[spec.key]
        name = f"rollout.{spec.key}"
        deps = (base_dep,)
        graph_nodes.append(
            Node(
                name=name,
                deps=deps,
                fn=_node_rollout_fn(node),
                enabled=_node_phase_enabled("rollout", node),
            )
        )
        rollout_node_names.append(name)

    deps = tuple(rollout_node_names) if rollout_node_names else ("rollout.td_prep",)
    graph_nodes.extend(
        [
            Node(
                name="rollout.reward_center",
                deps=deps,
                fn=_rollout_reward_center_fn(core),
                enabled=_phase_enabled("rollout"),
            ),
            Node(
                name="rollout.actions_check",
                deps=deps,
                fn=_rollout_actions_check_fn(core),
                enabled=_phase_enabled("rollout"),
            ),
            Node(
                name="rollout.send_actions",
                deps=("rollout.actions_check",),
                fn=_rollout_send_actions_fn(core),
                enabled=_phase_enabled("rollout"),
            ),
            Node(
                name="rollout.collect_infos",
                deps=("rollout.send_actions",),
                fn=_rollout_collect_infos_fn(),
                enabled=_phase_enabled("rollout"),
            ),
            Node(
                name="rollout.step_count",
                deps=("rollout.send_actions",),
                fn=_rollout_step_count_fn(),
                enabled=_phase_enabled("rollout"),
            ),
        ]
    )

    return graph_nodes


def _build_train_nodes(
    core: CoreTrainingLoop,
    nodes: dict[str, NodeBase],
    node_specs: list[NodeSpec],
) -> list[Node]:
    graph_nodes: list[Node] = [
        Node(
            name="train.sample_mb",
            fn=_train_sample_mb_fn(core),
            enabled=_phase_enabled("train"),
        ),
        Node(
            name="train.returns",
            deps=("train.sample_mb",),
            fn=_train_returns_fn(),
            enabled=_phase_enabled("train"),
        ),
        Node(
            name="train.policy_forward",
            deps=("train.returns",),
            fn=_train_policy_forward_fn(core),
            enabled=_phase_enabled("train"),
        ),
        Node(
            name="train.importance_ratio",
            deps=("train.policy_forward",),
            fn=_train_importance_ratio_fn(),
            enabled=_phase_enabled("train"),
        ),
        Node(
            name="train.advantages_pg",
            deps=("train.importance_ratio",),
            fn=_train_advantages_pg_fn(core),
            enabled=_phase_enabled("train"),
        ),
        Node(
            name="train.used_keys",
            deps=("train.policy_forward",),
            fn=_train_used_keys_fn(core),
            enabled=_phase_enabled("train"),
        ),
        Node(
            name="train.zero_grad",
            deps=("train.sample_mb",),
            fn=_train_zero_grad_fn(core),
            enabled=lambda _ctx, workspace: workspace["mb_idx"] % core.accumulate_minibatches == 0,
        ),
    ]

    train_node_names: list[str] = []
    base_dep = "train.advantages_pg"
    for spec in node_specs:
        if not spec.has_train:
            continue
        node = nodes[spec.key]
        name = f"train.{spec.key}"
        deps = [base_dep]
        if spec.has_rollout:
            deps.append(f"rollout.{spec.key}")
        graph_nodes.append(
            Node(
                name=name,
                deps=tuple(deps),
                fn=_node_train_fn(node),
                enabled=_node_phase_enabled("train", node),
            )
        )
        train_node_names.append(name)

    deps = tuple(train_node_names) if train_node_names else ("train.advantages_pg",)
    graph_nodes.extend(
        [
            Node(
                name="train.loss_sum",
                deps=deps,
                fn=_train_loss_sum_fn(core),
                enabled=_train_continue_enabled,
            ),
            Node(
                name="train.dummy_loss",
                deps=("train.loss_sum", "train.used_keys"),
                fn=_train_dummy_loss_fn(core),
                enabled=_train_continue_enabled,
            ),
            Node(
                name="train.backward",
                deps=("train.dummy_loss", "train.zero_grad"),
                fn=_train_backward_fn(),
                enabled=_train_continue_enabled,
            ),
            Node(
                name="train.optimizer_step",
                deps=("train.backward",),
                fn=_train_optimizer_step_fn(core),
                enabled=_train_continue_enabled,
            ),
            Node(
                name="train.on_mb_end",
                deps=("train.backward",),
                fn=_train_on_mb_end_fn(core),
                enabled=_train_continue_enabled,
            ),
        ]
    )

    return graph_nodes


def _rollout_env_wait_node(core: CoreTrainingLoop) -> Node:
    def _fn(context: ComponentContext, workspace: dict[str, Any]) -> dict[str, Any]:
        env = workspace["env"]
        with context.stopwatch("_rollout.env_wait"):
            o, r, d, t, ta, info, training_env_id, _, num_steps = env.get_observations()
        workspace["obs"] = o
        workspace["rewards"] = r
        workspace["dones"] = d
        workspace["truncateds"] = t
        workspace["teacher_actions"] = ta
        workspace["info"] = info
        workspace["training_env_id"] = training_env_id
        workspace["num_steps"] = num_steps
        workspace["last_env_id"] = training_env_id
        return {}

    return Node(name="rollout.env_wait", fn=_fn, enabled=_phase_enabled("rollout"))


def _rollout_td_prep_node(core: CoreTrainingLoop) -> Node:
    def _fn(context: ComponentContext, workspace: dict[str, Any]) -> dict[str, Any]:
        with context.stopwatch("_rollout.td_prep"):
            training_env_id = workspace["training_env_id"]
            td, agent_ids, baseline = _prepare_rollout_td(core, context, workspace)
            context.training_env_id = training_env_id
            workspace["td"] = td
            workspace["agent_ids"] = agent_ids
            workspace["baseline"] = baseline
        return {}

    return Node(
        name="rollout.td_prep",
        deps=("rollout.env_wait",),
        fn=_fn,
        enabled=_phase_enabled("rollout"),
    )


def _prepare_rollout_td(
    core: CoreTrainingLoop,
    context: ComponentContext,
    workspace: dict[str, Any],
) -> tuple[TensorDict, torch.Tensor, torch.Tensor]:
    training_env_id = workspace["training_env_id"]
    buffer_step = workspace["buffer_step"]
    td = buffer_step[training_env_id].clone()
    target_device = td.device
    td["env_obs"] = workspace["obs"].to(device=target_device, non_blocking=True)

    rewards = workspace["rewards"].to(device=target_device, non_blocking=True)
    td["rewards"] = rewards
    agent_ids = core._env_index_cache[training_env_id]
    td["training_env_ids"] = agent_ids.unsqueeze(1)

    avg_reward = context.state.avg_reward
    baseline = avg_reward[agent_ids]
    td["reward_baseline"] = baseline

    if target_device.type == "mps":
        td["dones"] = workspace["dones"].to(dtype=torch.float32).to(device=target_device, non_blocking=False)
        td["truncateds"] = workspace["truncateds"].to(dtype=torch.float32).to(device=target_device, non_blocking=False)
    else:
        td["dones"] = workspace["dones"].to(device=target_device, dtype=torch.float32, non_blocking=True)
        td["truncateds"] = workspace["truncateds"].to(
            device=target_device,
            dtype=torch.float32,
            non_blocking=True,
        )
    td["teacher_actions"] = workspace["teacher_actions"].to(device=target_device, dtype=torch.long, non_blocking=True)

    row_ids = core.experience.row_slot_ids[training_env_id]
    t_in_row = core.experience.t_in_row[training_env_id]
    td["row_id"] = row_ids
    td["t_in_row"] = t_in_row
    core.add_last_action_to_td(td)
    ensure_sequence_metadata(td, batch_size=td.batch_size.numel(), time_steps=1)
    return td, agent_ids, baseline


def _node_rollout_fn(node: NodeBase):
    def _fn(context: ComponentContext, workspace: dict[str, Any]) -> dict[str, Any]:
        with context.stopwatch("_rollout.inference"):
            node.rollout(workspace["td"], context)
        td = workspace["td"]
        if "actions" in td:
            workspace["actions_candidate"] = td["actions"]
        return {}

    return _fn


def _rollout_reward_center_fn(core: CoreTrainingLoop):
    def _fn(context: ComponentContext, workspace: dict[str, Any]) -> dict[str, Any]:
        avg_reward = context.state.avg_reward
        agent_ids = workspace["agent_ids"]
        baseline = workspace["baseline"]
        beta = float(context.config.advantage.reward_centering.beta)
        with torch.no_grad():
            rewards_f32 = workspace["td"]["rewards"].to(dtype=torch.float32)
            avg_reward[agent_ids] = baseline + beta * (rewards_f32 - baseline)
        context.state.avg_reward = avg_reward
        return {}

    return _fn


def _rollout_actions_check_fn(core: CoreTrainingLoop):
    def _fn(context: ComponentContext, workspace: dict[str, Any]) -> dict[str, Any]:
        td = workspace["td"]
        training_env_id = workspace["training_env_id"]
        if "actions_candidate" in workspace:
            td["actions"] = workspace["actions_candidate"]
        if "actions" not in td:
            raise RuntimeError("No node performed inference - at least one rollout node must generate actions")

        raw_actions = td["actions"].detach()
        if raw_actions.dim() != 1:
            raise ValueError(
                "Policies must emit a single discrete action id per agent; "
                f"received tensor of shape {tuple(raw_actions.shape)}"
            )

        actions_column = raw_actions.view(-1, 1)
        if core.last_action.device != actions_column.device:
            core.last_action = core.last_action.to(device=actions_column.device)
        if core.last_action.dtype != actions_column.dtype:
            actions_column = actions_column.to(dtype=core.last_action.dtype)

        target_buffer = core.last_action[training_env_id]
        if target_buffer.shape != actions_column.shape:
            msg = "last_action buffer shape mismatch: target=%s actions=%s raw=%s" % (
                target_buffer.shape,
                actions_column.shape,
                tuple(td["actions"].shape),
            )
            logger.error(msg, exc_info=True)
            raise RuntimeError(msg)

        target_buffer.copy_(actions_column)
        return {}

    return _fn


def _rollout_send_actions_fn(core: CoreTrainingLoop):
    def _fn(context: ComponentContext, workspace: dict[str, Any]) -> dict[str, Any]:
        env = workspace["env"]
        td = workspace["td"]
        with context.stopwatch("_rollout.send"):
            env.send_actions(td["actions"].cpu().numpy())
        return {}

    return _fn


def _rollout_collect_infos_fn():
    def _fn(context: ComponentContext, workspace: dict[str, Any]) -> dict[str, Any]:
        info = workspace.get("info")
        infos_list: list[dict[str, Any]] = list(info) if info else []
        if infos_list:
            workspace["raw_infos"].extend(infos_list)
        return {}

    return _fn


def _rollout_step_count_fn():
    def _fn(context: ComponentContext, workspace: dict[str, Any]) -> dict[str, Any]:
        workspace["total_steps"] += workspace["num_steps"]
        return {}

    return _fn


def _train_sample_mb_fn(core: CoreTrainingLoop):
    def _fn(context: ComponentContext, workspace: dict[str, Any]) -> dict[str, Any]:
        shared_loss_mb_data = core.experience.sample(
            mb_idx=workspace["mb_idx"],
            epoch=context.epoch,
            total_timesteps=context.config.total_timesteps,
            batch_size=context.config.batch_size,
            advantages=workspace["advantages_full"],
        )
        shared_loss_mb_data["advantages_full"] = NonTensorData(workspace["advantages_full"])
        workspace["shared_loss_data"] = shared_loss_mb_data
        workspace["loss_values"] = {}
        return {}

    return _fn


def _train_policy_forward_fn(core: CoreTrainingLoop):
    def _fn(context: ComponentContext, workspace: dict[str, Any]) -> dict[str, Any]:
        policy_td = workspace["shared_loss_data"]["sampled_mb"]
        policy_td = forward_policy_for_training(core.policy, policy_td, core.policy_spec)
        workspace["shared_loss_data"]["policy_td"] = policy_td
        return {}

    return _fn


def _train_importance_ratio_fn():
    def _fn(context: ComponentContext, workspace: dict[str, Any]) -> dict[str, Any]:
        shared = workspace["shared_loss_data"]
        sampled_mb = shared["sampled_mb"]
        policy_td = shared["policy_td"]
        if "act_log_prob" not in sampled_mb.keys() or "act_log_prob" not in policy_td.keys():
            return {}
        old_logprob = sampled_mb["act_log_prob"]
        new_logprob = policy_td["act_log_prob"].reshape(old_logprob.shape)
        logratio = torch.clamp(new_logprob - old_logprob, -10, 10)
        shared["importance_sampling_ratio"] = logratio.exp()
        return {}

    return _fn


def _train_returns_fn():
    def _fn(context: ComponentContext, workspace: dict[str, Any]) -> dict[str, Any]:
        shared = workspace["shared_loss_data"]
        sampled_mb = shared["sampled_mb"]
        if "values" not in sampled_mb.keys():
            return {}
        shared["sampled_mb"]["returns"] = shared["advantages"] + sampled_mb["values"]
        return {}

    return _fn


def _train_advantages_pg_fn(core: CoreTrainingLoop):
    def _fn(context: ComponentContext, workspace: dict[str, Any]) -> dict[str, Any]:
        shared = workspace["shared_loss_data"]
        sampled_mb = shared["sampled_mb"]
        advantage_method = workspace["advantage_method"]
        advantage_cfg = context.config.advantage

        if advantage_method == "delta_lambda":
            if "values" not in sampled_mb.keys():
                raise RuntimeError("delta_lambda advantages require minibatch['values']")

            new_values = shared["policy_td"]["values"]
            if new_values.dim() == 3 and new_values.shape[-1] == 1:
                new_values = new_values.squeeze(-1)
            new_values = new_values.reshape(sampled_mb["values"].shape)

            centered_rewards = sampled_mb["rewards"] - sampled_mb["reward_baseline"]
            shared["advantages_pg"] = compute_delta_lambda(
                values=new_values,
                rewards=centered_rewards,
                dones=sampled_mb["dones"],
                gamma=float(advantage_cfg.gamma),
                gae_lambda=float(advantage_cfg.gae_lambda),
            )
            return {}

        values_for_adv = sampled_mb["values"] if "values" in sampled_mb.keys() else None
        if values_for_adv is not None:
            if values_for_adv.dim() > 2:
                values_for_adv = values_for_adv.mean(dim=-1)

            importance_sampling_ratio = shared.get("importance_sampling_ratio", None)
            if importance_sampling_ratio is None:
                importance_sampling_ratio = torch.ones_like(values_for_adv)

            with torch.no_grad():
                centered_rewards = sampled_mb["rewards"] - sampled_mb["reward_baseline"]
                shared["advantages_pg"] = compute_advantage(
                    values_for_adv,
                    centered_rewards,
                    sampled_mb["dones"],
                    importance_sampling_ratio,
                    shared["advantages"].clone(),
                    advantage_cfg.gamma,
                    advantage_cfg.gae_lambda,
                    core.device,
                    advantage_cfg.vtrace_rho_clip,
                    advantage_cfg.vtrace_c_clip,
                )
        else:
            shared["advantages_pg"] = shared["advantages"]
        return {}

    return _fn


def _train_used_keys_fn(core: CoreTrainingLoop):
    def _fn(context: ComponentContext, workspace: dict[str, Any]) -> dict[str, Any]:
        used_keys: set[str] = set()
        policy_td = workspace["shared_loss_data"]["policy_td"]
        for name, node in core.nodes.items():
            spec = core.node_specs[name]
            if not spec.has_train:
                continue
            if not node.cfg.enabled or not node.node_gate_allows("train", context):
                continue
            used_keys.update(node.policy_output_keys(policy_td))
        workspace["used_keys"] = used_keys
        return {}

    return _fn


def _train_zero_grad_fn(core: CoreTrainingLoop):
    def _fn(context: ComponentContext, workspace: dict[str, Any]) -> dict[str, Any]:
        core.optimizer.zero_grad(set_to_none=True)
        return {}

    return _fn


def _node_train_fn(node: NodeBase):
    def _fn(context: ComponentContext, workspace: dict[str, Any]) -> dict[str, Any]:
        loss_val, shared, stop = node.train(workspace["shared_loss_data"], context, workspace["mb_idx"])
        workspace["shared_loss_data"] = shared
        workspace["loss_values"][node.node_name] = loss_val
        if stop:
            workspace["stop_update_epoch"] = True
        return {}

    return _fn


def _train_loss_sum_fn(core: CoreTrainingLoop):
    def _fn(context: ComponentContext, workspace: dict[str, Any]) -> dict[str, Any]:
        total = sum(
            workspace["loss_values"].values(),
            torch.tensor(0.0, dtype=torch.float32, device=core.device),
        )
        workspace["total_loss"] = total
        return {}

    return _fn


def _train_dummy_loss_fn(core: CoreTrainingLoop):
    def _fn(context: ComponentContext, workspace: dict[str, Any]) -> dict[str, Any]:
        shared = workspace["shared_loss_data"]
        total_loss = workspace["total_loss"]
        policy_td = shared["policy_td"]
        workspace["total_loss"] = add_dummy_loss_for_unused_params(
            total_loss, td=policy_td, used_keys=workspace["used_keys"]
        )
        return {}

    return _fn


def _train_backward_fn():
    def _fn(context: ComponentContext, workspace: dict[str, Any]) -> dict[str, Any]:
        workspace["total_loss"].backward()
        return {}

    return _fn


def _train_optimizer_step_fn(core: CoreTrainingLoop):
    def _fn(context: ComponentContext, workspace: dict[str, Any]) -> dict[str, Any]:
        mb_idx = workspace["mb_idx"]
        if (mb_idx + 1) % core.accumulate_minibatches != 0:
            return {}

        actual_max_grad_norm = float(workspace.get("max_grad_norm", 0.5))
        for spec in core.node_specs_order:
            if not spec.has_train:
                continue
            node = core.nodes.get(spec.key)
            if node is None:
                continue
            if not node.cfg.enabled or not node.node_gate_allows("train", context):
                continue
            if hasattr(node.cfg, "max_grad_norm"):
                actual_max_grad_norm = float(node.cfg.max_grad_norm)
                break

        torch.nn.utils.clip_grad_norm_(core.policy.parameters(), actual_max_grad_norm)
        core.optimizer.step()
        if core.device.type == "cuda":
            torch.cuda.synchronize()

        return {}

    return _fn


def _train_on_mb_end_fn(core: CoreTrainingLoop):
    def _fn(context: ComponentContext, workspace: dict[str, Any]) -> dict[str, Any]:
        for name, node in core.nodes.items():
            spec = core.node_specs[name]
            if spec.has_train:
                node.on_mb_end(context, workspace["mb_idx"])
        return {}

    return _fn


def _train_continue_enabled(context: ComponentContext, workspace: dict[str, Any]) -> bool:
    return _phase_enabled("train")(context, workspace) and not bool(workspace.get("stop_update_epoch", False))


def _phase_enabled(phase: str):
    def _enabled(context: ComponentContext, workspace: dict[str, Any]) -> bool:
        return workspace.get("phase") == phase

    return _enabled


def _node_phase_enabled(phase: str, node: NodeBase):
    def _enabled(context: ComponentContext, workspace: dict[str, Any]) -> bool:
        if workspace.get("phase") != phase:
            return False
        if not node.cfg.enabled:
            return False
        return node.node_gate_allows(phase, context)

    return _enabled
