"""Policy checkpoint management component."""

import logging
from pathlib import Path
from typing import Optional

import torch
from pydantic import Field
from safetensors.torch import load_file as load_safetensors_file

from metta.agent.policy import Policy, PolicyArchitecture
from metta.rl.checkpoint_manager import CheckpointManager
from metta.rl.training import DistributedHelper, TrainerComponent
from mettagrid.base_config import Config
from mettagrid.policy.loader import initialize_or_load_policy
from mettagrid.policy.policy_env_interface import PolicyEnvInterface
from mettagrid.util.module import load_symbol
from mettagrid.util.uri_resolvers.schemes import policy_spec_from_uri, resolve_uri

logger = logging.getLogger(__name__)


class CheckpointerConfig(Config):
    epoch_interval: int = Field(default=30, ge=0)


class Checkpointer(TrainerComponent):
    """Manages policy checkpointing with distributed awareness."""

    def __init__(
        self,
        *,
        config: CheckpointerConfig,
        checkpoint_manager: CheckpointManager,
        distributed_helper: DistributedHelper,
        policy_architecture: PolicyArchitecture,
    ) -> None:
        super().__init__(epoch_interval=max(1, config.epoch_interval))
        self._master_only = True
        self._config = config
        self._checkpoint_manager = checkpoint_manager
        self._distributed = distributed_helper
        self._policy_architecture: PolicyArchitecture = policy_architecture
        self._latest_policy_uri: Optional[str] = None

    def register(self, context) -> None:
        super().register(context)
        context.latest_policy_uri_fn = self.get_latest_policy_uri
        context.latest_policy_uri_value = self.get_latest_policy_uri()

    def load_or_create_policy(
        self,
        policy_env_info: PolicyEnvInterface,
        *,
        policy_uri: Optional[str] = None,
    ) -> Policy:
        """Load the latest policy checkpoint or create a new policy."""
        candidate_uri = policy_uri or self._checkpoint_manager.get_latest_checkpoint()
        load_device = torch.device(self._distributed.config.device)

        if self._distributed.is_distributed:
            normalized_uri = self._distributed.broadcast_from_master(
                resolve_uri(candidate_uri).canonical if self._distributed.is_master() and candidate_uri else None
            )

            if normalized_uri:
                payload: tuple[str, dict[str, object], dict[str, torch.Tensor]] | None = None
                if self._distributed.is_master():
                    policy_spec = policy_spec_from_uri(normalized_uri)
                    state_dict = load_safetensors_file(str(Path(policy_spec.data_path).expanduser()))
                    payload = (
                        policy_spec.class_path,
                        policy_spec.init_kwargs or {},
                        {k: v.cpu() for k, v in state_dict.items()},
                    )
                payload = self._distributed.broadcast_from_master(payload)
                class_path, init_kwargs, state_dict = payload
                init_kwargs = dict(init_kwargs)
                if "device" in init_kwargs:
                    init_kwargs["device"] = str(load_device)
                policy_class = load_symbol(class_path)
                policy = policy_class(policy_env_info, **init_kwargs)  # type: ignore[call-arg]
                if hasattr(policy, "to"):
                    policy = policy.to(load_device)
                policy.load_state_dict(state_dict, strict=True)
                initialize = getattr(policy, "initialize_to_environment", None)
                if callable(initialize):
                    initialize(policy_env_info, load_device)

                if self._distributed.is_master():
                    self._latest_policy_uri = normalized_uri
                    logger.info("Loaded policy from %s", normalized_uri)
                return policy

        if candidate_uri:
            policy = initialize_or_load_policy(
                policy_env_info,
                policy_spec_from_uri(candidate_uri),
                device_override=str(load_device),
            )
            self._latest_policy_uri = resolve_uri(candidate_uri).canonical
            logger.info("Loaded policy from %s", candidate_uri)
            return policy

        logger.info("Creating new policy for training run")
        return self._policy_architecture.make_policy(policy_env_info)

    def get_latest_policy_uri(self) -> Optional[str]:
        return self._checkpoint_manager.get_latest_checkpoint() or self._latest_policy_uri

    def on_epoch_end(self, epoch: int) -> None:
        if not self._distributed.should_checkpoint():
            return
        self._save_policy(epoch)

    def on_training_complete(self) -> None:
        if not self._distributed.should_checkpoint():
            return
        self._save_policy(self.context.epoch)

    def _save_policy(self, epoch: int) -> None:
        slot_policies = getattr(self.context, "slot_policies", None) or {}
        slot_cfgs = list(getattr(self.context, "policy_slots", []) or [])
        slot_lookup = getattr(self.context, "slot_id_lookup", {})

        if slot_policies and slot_cfgs:
            latest_uris: dict[str, str] = {}
            for slot_cfg, policy, arch_spec, checkpoint_slot_id in self._iter_checkpoint_slots(
                slot_cfgs, slot_lookup, slot_policies
            ):
                uri = self._checkpoint_manager.save_policy_checkpoint(
                    state_dict=policy.state_dict(),
                    architecture=arch_spec,
                    epoch=epoch,
                    slot_id=checkpoint_slot_id,
                )
                latest_uris[slot_cfg.id] = uri

            if latest_uris:
                self._update_latest_uris(latest_uris, slot_cfgs, epoch)
        else:
            uri = self._checkpoint_manager.save_policy_checkpoint(
                state_dict=self.context.policy.state_dict(),
                architecture=self._policy_architecture,
                epoch=epoch,
            )

            self._latest_policy_uri = uri
            self.context.latest_policy_uri_value = uri
            self.context.latest_saved_policy_epoch = epoch

        # Log latest checkpoint URI to wandb if available
        stats_reporter = getattr(self.context, "stats_reporter", None)
        wandb_run = getattr(stats_reporter, "wandb_run", None) if stats_reporter is not None else None
        if wandb_run is not None:
            payload = {
                "checkpoint/latest_epoch": float(epoch),
            }
            if self._latest_policy_uri:
                payload["checkpoint/latest_uri"] = self._latest_policy_uri
            latest_uris = getattr(self.context, "latest_policy_uris", None) or {}
            for slot_id, uri in latest_uris.items():
                payload[f"checkpoint/latest_uri/{slot_id}"] = uri
            wandb_run.log(payload, step=self.context.agent_step)
            logger.info("Logged checkpoint URIs to wandb")

    def _iter_checkpoint_slots(
        self,
        slot_cfgs,
        slot_lookup: dict[str, int],
        slot_policies: dict[int, Policy],
    ):
        for slot_cfg in slot_cfgs:
            if not slot_cfg.trainable:
                continue
            slot_idx = slot_lookup.get(slot_cfg.id)
            if slot_idx is None:
                continue
            policy = slot_policies.get(slot_idx)
            if policy is None:
                continue
            arch_spec = self._resolve_architecture_spec(
                policy,
                slot_cfg,
                fallback=self._policy_architecture if slot_cfg.use_trainer_policy else None,
            )
            if not arch_spec:
                logger.warning("Skipping checkpoint for slot '%s' (no architecture spec available)", slot_cfg.id)
                continue
            checkpoint_slot_id = None if slot_cfg.use_trainer_policy else slot_cfg.id
            yield slot_cfg, policy, arch_spec, checkpoint_slot_id

    def _update_latest_uris(self, latest_uris: dict[str, str], slot_cfgs, epoch: int) -> None:
        self.context.latest_policy_uris = latest_uris
        main_uri = None
        for slot_cfg in slot_cfgs:
            if slot_cfg.use_trainer_policy and slot_cfg.id in latest_uris:
                main_uri = latest_uris[slot_cfg.id]
                break
        if main_uri is None and slot_cfgs and slot_cfgs[0].id in latest_uris:
            main_uri = latest_uris[slot_cfgs[0].id]
        if main_uri:
            self._latest_policy_uri = main_uri
            self.context.latest_policy_uri_value = main_uri
            self.context.latest_saved_policy_epoch = epoch

    def _resolve_architecture_spec(
        self,
        policy: Policy,
        slot_cfg,
        *,
        fallback: PolicyArchitecture | None = None,
    ) -> str | None:
        if slot_cfg and getattr(slot_cfg, "architecture_spec", None):
            return slot_cfg.architecture_spec
        spec = getattr(policy, "architecture_spec", None)
        if isinstance(spec, str):
            return spec
        cfg = getattr(policy, "config", None)
        if isinstance(cfg, PolicyArchitecture):
            return cfg.to_spec()
        if fallback is not None:
            return fallback.to_spec()
        return None
