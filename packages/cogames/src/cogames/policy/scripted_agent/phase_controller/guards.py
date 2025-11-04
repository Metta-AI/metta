"""Guard predicates used by the scripted agent phase controller."""

from __future__ import annotations

from typing import Optional

from cogames.policy.scripted_agent.state import AgentState

from .controller import Context, GamePhase, Guard


def has_all_materials(state, ctx: Context) -> bool:
    reqs = getattr(state, "heart_requirements", None)
    if reqs:
        germ_needed = max(1, reqs.get("germanium", 1))
        silicon_req = reqs.get("silicon", 0)
        carbon_req = reqs.get("carbon", 0)
        oxygen_req = reqs.get("oxygen", 0)
        energy_req = reqs.get("energy", 20)
    else:
        germ_needed = 5 if state.hearts_assembled == 0 else max(2, 5 - state.hearts_assembled)
        silicon_req = 50
        carbon_req = 20
        oxygen_req = 20
        energy_req = 20

    return (
        state.germanium >= germ_needed
        and state.silicon >= silicon_req
        and state.carbon >= carbon_req
        and state.oxygen >= oxygen_req
        and state.energy >= energy_req
    )


def low_energy(state, ctx: Context) -> bool:
    map_size = max(ctx.env.c_env.map_width, ctx.env.c_env.map_height)
    threshold = (
        ctx.policy_impl.hyperparams.recharge_start_small
        if map_size < 50
        else ctx.policy_impl.hyperparams.recharge_start_large
    )
    return state.energy < threshold


def recharged_enough(state, ctx: Context) -> bool:
    policy_impl = getattr(ctx, "policy_impl", None)
    tolerance = getattr(policy_impl, "recharge_idle_tolerance", 3)
    if getattr(state, "recharge_ticks_without_gain", 0) >= tolerance:
        return True
    if policy_impl is not None and getattr(state, "energy", 0) >= policy_impl.RECHARGE_STOP:
        return True
    return False


def carrying_heart(state, ctx: Context) -> bool:  # noqa: ARG001
    return state.heart > 0


def have_assembler_discovered(state, ctx: Context) -> bool:
    if getattr(state, "assembler_discovered", False):
        return True
    policy_impl = getattr(ctx, "policy_impl", None)
    if policy_impl is None:
        return False
    return "assembler" in getattr(policy_impl, "_station_positions", {})


def blocked_by_clipped(state, ctx: Context) -> bool:  # noqa: ARG001
    return state.blocked_by_clipped_extractor is not None


def get_blocked_extractor_resource_type(state, ctx: Context) -> Optional[str]:
    if state.blocked_by_clipped_extractor is None:
        return None

    policy_impl = getattr(state, "_policy_impl", None) or getattr(ctx, "policy_impl", None)
    if policy_impl is None:
        return None

    extractor = policy_impl.extractor_memory.get_at_position(state.blocked_by_clipped_extractor)
    return extractor.resource_type if extractor else None


def need_craft_resource_for_blocked(state, ctx: Context) -> bool:
    resource_type = get_blocked_extractor_resource_type(state, ctx)
    if resource_type is None:
        return False

    recipes = (
        state.unclip_recipes
        if getattr(state, "unclip_recipes", None)
        else {
            "oxygen": "carbon",
            "carbon": "oxygen",
            "germanium": "silicon",
            "silicon": "germanium",
        }
    )
    craft_resource = recipes.get(resource_type)
    if craft_resource is None:
        return False
    return getattr(state, craft_resource, 0) < 1


def have_chest_discovered(state, ctx: Context) -> bool:
    if getattr(state, "chest_discovered", False):
        return True
    policy_impl = getattr(ctx, "policy_impl", None)
    if policy_impl is None:
        return False
    return "chest" in getattr(policy_impl, "_station_positions", {})


def decoder_ready_for_unclipping(state, ctx: Context) -> bool:
    if state.blocked_by_clipped_extractor is None:
        return False

    resource_type = get_blocked_extractor_resource_type(state, ctx)
    if resource_type is None:
        return False

    item_map = {
        "oxygen": "decoder",
        "carbon": "modulator",
        "germanium": "resonator",
        "silicon": "scrambler",
    }
    item_name = item_map.get(resource_type)
    if item_name is None:
        return False
    return getattr(state, item_name, 0) > 0


def progress_stalled(max_steps: int) -> Guard:
    def _guard(state, ctx: Context) -> bool:
        pinv = state.phase_entry_inventory or {}
        now = {"g": state.germanium, "si": state.silicon, "c": state.carbon, "o": state.oxygen}
        no_delta = all(
            now.get(k, 0) <= pinv.get(mapping, 0)
            for k, mapping in [("g", "germanium"), ("si", "silicon"), ("c", "carbon"), ("o", "oxygen")]
        )
        return no_delta and (ctx.step - state.phase_entry_step) >= max_steps

    return _guard


def no_extractors_available(phase: GamePhase) -> Guard:
    def _guard(state, ctx: Context) -> bool:
        policy_impl = getattr(ctx, "policy_impl", None)
        if policy_impl is None or not getattr(policy_impl, "extractor_memory", None):
            return False
        resource_map = {
            GamePhase.GATHER_GERMANIUM: "germanium",
            GamePhase.GATHER_SILICON: "silicon",
            GamePhase.GATHER_CARBON: "carbon",
            GamePhase.GATHER_OXYGEN: "oxygen",
            GamePhase.RECHARGE: "charger",
        }
        resource = resource_map.get(phase)
        if resource is None:
            return False
        extractors = policy_impl.extractor_memory.get_by_type(resource)
        if not extractors:
            return True
        available = [
            e for e in extractors if not e.is_depleted() and e.is_available(ctx.step, policy_impl.cooldown_remaining)
        ]
        return len(available) == 0

    return _guard


def assemble_slot_available(state: AgentState, ctx: Context) -> bool:
    policy_impl = getattr(ctx, "policy_impl", None)
    if policy_impl is None or not hasattr(policy_impl, "can_agent_reserve_assembler"):
        return True
    try:
        return bool(policy_impl.can_agent_reserve_assembler(state))
    except Exception:
        return True


def have_charger_discovered(state, ctx: Context) -> bool:  # noqa: ARG001
    policy_impl = getattr(ctx, "policy_impl", None)
    if policy_impl is None or not getattr(policy_impl, "extractor_memory", None):
        return False
    return bool(policy_impl.extractor_memory.get_by_type("charger"))


def should_recharge(ctx: Context, s: AgentState) -> bool:
    policy = ctx.policy_impl
    width = ctx.env.game.map_builder.width if hasattr(ctx.env.game.map_builder, "width") else 50
    threshold = policy.hyperparams.recharge_start_small if width < 50 else policy.hyperparams.recharge_start_large
    return s.energy <= threshold


def should_keep_recharging(ctx: Context, s: AgentState) -> bool:
    policy = ctx.policy_impl
    width = ctx.env.game.map_builder.width if hasattr(ctx.env.game.map_builder, "width") else 50
    stop_threshold = policy.hyperparams.recharge_stop_small if width < 50 else policy.hyperparams.recharge_stop_large
    if policy.hyperparams.recharge_until_full:
        return s.energy < stop_threshold
    return s.energy < stop_threshold and not s.last_attempt_was_use


def energy_low(ctx: Context, s: AgentState) -> bool:
    return should_recharge(ctx, s)


def recharge_complete(ctx: Context, s: AgentState) -> bool:
    return not should_keep_recharging(ctx, s)


__all__ = [
    "has_all_materials",
    "low_energy",
    "recharged_enough",
    "carrying_heart",
    "have_assembler_discovered",
    "blocked_by_clipped",
    "get_blocked_extractor_resource_type",
    "need_craft_resource_for_blocked",
    "have_chest_discovered",
    "decoder_ready_for_unclipping",
    "progress_stalled",
    "no_extractors_available",
    "have_charger_discovered",
]
