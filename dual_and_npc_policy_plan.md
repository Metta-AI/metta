# Dual & NPC Policy Plan (Metta)

## Goals

- Make multiple agent policies first-class for training and evaluation (scripted + neural).
- Support dual-policy (two neural nets) for training and evaluation without legacy PPO.
- Keep split-loss stack (ppo_actor/ppo_critic, sliced losses) and integrate masking cleanly.
- Use the leanest model: slot table + agent-slot map.

## Milestones

1. Slot schema + agent-slot map + TD annotations + registry (no dual training yet).
2. Loss filtering via `loss_profile_id`/`is_trainable` in split PPO and sliced losses.
3. End-to-end dual-policy training with frozen NPC neural policy.
4. Dual-policy evaluation with per-slot metrics (scripted + neural mixes).
5. Optional co-training mode (two trainable nets) + optimizer scheduling.

## Phase 0 – Foundations & API Shape (detailed, lean variant)

### Outputs

- `PolicySlot` schema and validation.
- `agent_slot_map` support in trainer/eval/simulation configs.
- `SlotRegistry` for loading/caching policies (URI or class path).
- TD/experience annotations: `slot_id`, `loss_profile_id`, `is_trainable_agent`.
- Backward-compatible defaults (single slot mapped to all agents).

### Work items

- **Config surface**: Add `trainer_cfg.policy_slots: list[PolicySlot]`. Fields: `id`, `policy_uri|class_path`,
  `policy_kwargs`, `trainable: bool`, optional `loss_profile`, optional `device`. Validation: unique ids; at least one
  trainable slot for training; slots referenced by the map must exist.
- **Agent-slot map**: Add `trainer_cfg.agent_slot_map: list[str|int]` of length `num_agents` (or per-env override)
  mapping each agent index to a slot id. Default: all agents → first slot. This single map enables kickstarter slicing
  by assigning some agents to a teacher slot.
- **Loss profiles (config only)**: Introduce `loss_profiles` map: name → set of losses to run. Phase 0 only tags rows
  with `loss_profile_id`; execution gating comes in Phase 2.
- **Registry**: `SlotRegistry.get(slot, policy_env_info, device) -> Policy` loads scripted (class path) or NN (URI).
  Caches by descriptor; calls `initialize_to_environment`.
- **TD annotations**: During rollout, set per-row `slot_id`, `loss_profile_id`, `is_trainable_agent`. Add these to
  experience spec.
- **Multi-agent forward**: Use existing `Policy`/`MultiAgentPolicy` orchestration: group rows by slot_id, forward each
  policy once, merge actions/logprobs back.
- **Env metadata**: Allow providing `agent_slot_map` via simulation/recipe; default infer single slot. Ensure
  compatibility with mettagrid Puffer env agent counts.
- **Backward compatibility**: If `policy_slots` absent, synthesize one slot (`main`) pointing to the configured policy;
  map all agents to it; losses behave unchanged.
- **Docs**: Short example config: two slots (student trainable, teacher scripted), slot map
  `[teacher, teacher, student, student]`.

### Acceptance criteria

- Training without new config is unchanged.
- With two slots (one scripted, one trainable) and a slot map, rollout completes; TD/replay contain `slot_id`,
  `loss_profile_id`, `is_trainable_agent`.
- No reliance on legacy monolithic PPO.

### Loss groundwork

- Carry `loss_profile_id` in TD so sliced/kickstarter can later select rows without bespoke masks. For now, defaults map
  all rows to a single profile.

## Phase 1 – Training Pipeline (updated)

- Rollout: multi-slot forward + TD annotations; cached masks per slot from `agent_slot_map`.
- Replay spec includes slot/loss profile flags.
- Loss base adds a shared filter: minibatch rows restricted by `loss_profile_id` and `is_trainable_agent` when the loss
  is trainable-only.

## Phase 2 – Loss Execution & Masking

- Implement the shared filter in loss base; opt-in per loss (split PPO, sliced_kickstarter, sliced_scripted_cloner,
  etc.).
- Loss profiles: wire config so each profile declares which losses run; rows with a profile skip losses not in their
  profile.
- Preserve #4107 behavior: kickstarter slices remain possible by choosing maps where some agents map to teacher slots
  with a profile that includes supervised loss, others to PPO-only.

## Phase 3 – Dual-Policy Training

- Two slots pointing to two NN checkpoints; slot map assigns agents. One slot may be `trainable=False` (frozen NPC) or
  both trainable (co-training). Gradients flow only from rows where `is_trainable_agent=True`.
- Optional alternate optimizer steps per binding (config knob).

## Phase 4 – Evaluation & Simulation

- `simulation_config`/`eval_config` accept `policy_slots` + `agent_slot_map`. Runner loads slots via registry, assigns
  control per agent, and reports metrics grouped by `slot_id`.
- Supports head-to-head or mixed scripted/NN eval without new abstractions.

## Phase 5 – Cogames/Scripted Integration

- Scripted policies load via `class_path` through the registry. Puffer teacher actions map to a teacher slot; no special
  slot code.

## Phase 6 – Canonicalization (no legacy path)

- Slots + agent_slot_map are the only control surface; legacy shorthands like `dual_policy` are out.
- Split-loss stack is the default; legacy combined PPO is removed.
- Defaults still auto-create a single slot for convenience.

## Phase 7 – Testing

- Unit: registry loads, slot-map validation, TD annotation shapes, loss filtering logic.
- Integration: rollout with two slots (scripted + trainable) verifying action merge and masks; kickstarter-style map.
- E2E: recipe training with slot map + eval vs scripted; CI on CPU.
