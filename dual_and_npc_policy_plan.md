# Dual & NPC Policy Plan (Metta)

## Goals

- Make multiple agent policies first-class for training and evaluation (scripted + neural).
- Support dual-policy (two neural nets) for training and evaluation without legacy PPO.
- Keep split-loss stack (ppo_actor/ppo_critic, sliced losses) and integrate masking cleanly.
- Use the leanest model: binding table + agent-binding map; no extra slot hierarchy.

## Milestones

1. Binding schema + agent-binding map + TD annotations + registry (no dual training yet).
2. Loss filtering via `loss_profile_id`/`is_trainable` in split PPO and sliced losses.
3. End-to-end dual-policy training with frozen NPC neural policy.
4. Dual-policy evaluation with per-binding metrics (scripted + neural mixes).
5. Optional co-training mode (two trainable nets) + optimizer scheduling.

## Phase 0 – Foundations & API Shape (detailed, lean variant)

### Outputs

- `PolicyBinding` schema and validation.
- `agent_binding_map` support in trainer/eval/simulation configs.
- `PolicyRegistry` for loading/caching policies (URI or class path).
- TD/experience annotations: `binding_id`, `loss_profile_id`, `is_trainable_agent`.
- Backward-compatible defaults (single binding mapped to all agents).

### Work items

- **Config surface**: Add `trainer_cfg.policy_bindings: list[PolicyBinding]`. Fields: `id`, `policy_uri|class_path`,
  `policy_kwargs`, `trainable: bool`, optional `loss_profile`, optional `device`. Validation: unique ids; at least one
  trainable binding for training; bindings referenced by the map must exist.
- **Agent-binding map**: Add `trainer_cfg.agent_binding_map: list[str|int]` of length `num_agents` (or per-env override)
  mapping each agent index to a binding id. Default: all agents → first binding. This single map replaces “slots” and
  enables kickstarter slicing by assigning some agents to a teacher binding.
- **Loss profiles (config only)**: Introduce `loss_profiles` map: name → set of losses to run. Phase 0 only tags rows
  with `loss_profile_id`; execution gating comes in Phase 2.
- **Registry**: `PolicyRegistry.get(binding, policy_env_info, device) -> Policy` loads scripted (class path) or NN
  (URI). Caches by descriptor; calls `initialize_to_environment`.
- **TD annotations**: During rollout, set per-row `binding_id`, `loss_profile_id`, `is_trainable_agent`. Add these to
  experience spec.
- **Multi-agent forward**: Use existing `Policy`/`MultiAgentPolicy` orchestration: group rows by binding_id, forward
  each policy once, merge actions/logprobs back.
- **Env metadata**: Allow providing `agent_binding_map` via simulation/recipe; default infer single binding. Ensure
  compatibility with mettagrid Puffer env agent counts.
- **Backward compatibility**: If `policy_bindings` absent, synthesize one binding (`main`) pointing to the configured
  policy; map all agents to it; losses behave unchanged.
- **Docs**: Short example config: two bindings (student trainable, teacher scripted), binding map
  `[teacher, teacher, student, student]`.

### Acceptance criteria

- Training without new config is unchanged.
- With two bindings (one scripted, one trainable) and a binding map, rollout completes; TD/replay contain `binding_id`,
  `loss_profile_id`, `is_trainable_agent`.
- No reliance on legacy monolithic PPO.

### Loss groundwork

- Carry `loss_profile_id` in TD so sliced/kickstarter can later select rows without bespoke masks. For now, defaults map
  all rows to a single profile.

## Phase 1 – Training Pipeline (updated)

- Rollout: multi-binding forward + TD annotations; cached masks per binding from `agent_binding_map`.
- Replay spec includes binding/loss profile flags.
- Loss base adds a shared filter: minibatch rows restricted by `loss_profile_id` and `is_trainable_agent` when the loss
  is trainable-only.

## Phase 2 – Loss Execution & Masking

- Implement the shared filter in loss base; opt-in per loss (split PPO, sliced_kickstarter, sliced_scripted_cloner,
  etc.).
- Loss profiles: wire config so each profile declares which losses run; rows with a profile skip losses not in their
  profile.
- Preserve #4107 behavior: kickstarter slices remain possible by choosing maps where some agents map to teacher binding
  with a profile that includes supervised loss, others to PPO-only.

## Phase 3 – Dual-Policy Training

- Two bindings pointing to two NN checkpoints; binding map assigns agents. One binding may be `trainable=False` (frozen
  NPC) or both trainable (co-training). Gradients flow only from rows where `is_trainable_agent=True`.
- Optional alternate optimizer steps per binding (config knob).

## Phase 4 – Evaluation & Simulation

- `simulation_config`/`eval_config` accept `policy_bindings` + `agent_binding_map`. Runner loads bindings via registry,
  assigns control per agent, and reports metrics grouped by `binding_id`.
- Supports head-to-head or mixed scripted/NN eval without new abstractions.

## Phase 5 – Cogames/Scripted Integration

- Scripted policies load via `class_path` through the registry. Puffer teacher actions map to a teacher binding; no
  special slot code.

## Phase 6 – Canonicalization (no legacy path)

- Bindings + agent_binding_map are the only control surface; legacy shorthands like `dual_policy` are out.
- Split-loss stack is the default; legacy combined PPO is removed.
- Defaults still auto-create a single binding for convenience.

## Phase 7 – Testing

- Unit: registry loads, binding-map validation, TD annotation shapes, loss filtering logic.
- Integration: rollout with two bindings (scripted + trainable) verifying action merge and masks; kickstarter-style map.
- E2E: recipe training with binding map + eval vs scripted; CI on CPU.
