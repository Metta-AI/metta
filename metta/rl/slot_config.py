"""Configuration helpers for policy slots and loss profiles."""

from typing import Any, Dict, List, Optional, Sequence

from pydantic import Field, model_validator

from mettagrid.base_config import Config


class LossProfileConfig(Config):
    """Names the losses that should run for agents attached to this profile."""

    losses: List[str] = Field(default_factory=list)


class PolicySlotConfig(Config):
    """Associates a policy loader with metadata used during rollout and training."""

    id: str = Field(description="Unique slot identifier")
    policy_uri: Optional[str] = Field(default=None, description="Checkpoint URI for neural policies")
    class_path: Optional[str] = Field(default=None, description="Import path for scripted policies")
    policy_kwargs: Dict[str, Any] = Field(default_factory=dict)
    trainable: bool = Field(default=True, description="Whether gradients should flow for this slot")
    loss_profile: Optional[str] = Field(default=None, description="Optional loss profile name for this slot")
    use_trainer_policy: bool = Field(
        default=False,
        description="If True, reuse the trainer-provided policy instance instead of loading a new one.",
    )

    @model_validator(mode="after")
    def validate_loader(self) -> "PolicySlotConfig":
        if not self.use_trainer_policy and not (self.policy_uri or self.class_path):
            raise ValueError("policy_uri or class_path must be set unless use_trainer_policy=True")
        if self.use_trainer_policy and (self.policy_uri or self.class_path):
            raise ValueError("use_trainer_policy=True is mutually exclusive with policy_uri/class_path")
        return self


def resolve_policy_slots(
    slots_cfg: Sequence[PolicySlotConfig] | None,
    *,
    num_agents: int,
    agent_slot_map: Sequence[str] | None,
    ensure_trainer_slot: bool,
) -> tuple[list[PolicySlotConfig], list[str], list[int]]:
    slots = list(slots_cfg or [])
    if not slots:
        if ensure_trainer_slot:
            slots = [PolicySlotConfig(id="main", use_trainer_policy=True, trainable=True)]
        else:
            raise ValueError("policy_slots must be provided when ensure_trainer_slot=False")

    trainer_slot_idx = next((idx for idx, slot in enumerate(slots) if slot.use_trainer_policy), None)
    if ensure_trainer_slot:
        if trainer_slot_idx is None:
            slots.insert(0, PolicySlotConfig(id="main", use_trainer_policy=True, trainable=True))
        elif trainer_slot_idx != 0:
            slots.insert(0, slots.pop(trainer_slot_idx))
        if sum(1 for slot in slots if slot.use_trainer_policy) > 1:
            raise ValueError("Only one slot may set use_trainer_policy=True")

    slot_lookup: dict[str, int] = {}
    for slot in slots:
        if slot.id in slot_lookup:
            raise ValueError(f"Duplicate policy slot id '{slot.id}'")
        slot_lookup[slot.id] = len(slot_lookup)

    resolved_agent_map = list(agent_slot_map) if agent_slot_map is not None else [slots[0].id for _ in range(num_agents)]
    if len(resolved_agent_map) != num_agents:
        raise ValueError(f"agent_slot_map must have length num_agents ({num_agents}); got {len(resolved_agent_map)}")

    slot_ids: list[int] = []
    for idx, slot_id in enumerate(resolved_agent_map):
        if slot_id not in slot_lookup:
            raise ValueError(f"agent_slot_map[{idx}] references unknown slot id '{slot_id}'")
        slot_ids.append(slot_lookup[slot_id])

    return slots, resolved_agent_map, slot_ids
