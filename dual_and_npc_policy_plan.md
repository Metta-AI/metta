# Dual & NPC Policy Plan (Metta)

## Goals
- Make multiple agent policies first-class for training and evaluation (scripted + neural).
- Support dual-policy (two neural nets) for training and evaluation without legacy PPO.
- Keep split-loss stack (ppo_actor/ppo_critic, sliced losses) and integrate masks cleanly.

## Milestones
1. AgentSlot config + registry + slot masks in rollout/experience (no dual training yet).
2. Loss masking in split PPO + sliced losses using `is_trainable_agent`/slot masks.
3. End-to-end dual-policy training with frozen NPC neural policy.
4. Dual-policy evaluation with per-slot metrics (scripted + neural mixes).
5. Optional co-training mode (two trainable nets) + optimizer scheduling.

## Phase 0 – Foundations & API Shape (detailed)
### Outputs
- `AgentSlotConfig` schema and validation.
- `AgentRegistry` service for loading/caching policies (URI or class path).
- Policy/env metadata carries per-agent slot mapping.
- Backward-compatible defaults (single student slot).

### Work items
- **Config surface**: Add `trainer_cfg.agent_slots: list[AgentSlotConfig]`. Fields: `name`, `policy_uri|class_path`, `policy_kwargs`, `role` (`student|teacher|npc|eval_only`), `trainable: bool`, `priority` (tie-break), optional `device`. Validation: at least one slot marked trainable in training; unique names; exactly `num_agents` covered by masks at runtime. Add optional `loss_profile` ref (see below) so slots can opt into specific loss stacks (e.g., kickstarter vs PPO).
- **Shorthand**: Keep `dual_policy.enabled` as an expander that produces two slots (`student`, `npc`) to ease migration; marked deprecated once slots are stable.
- **AgentRegistry**: Single entry point to instantiate scripted (class path) or neural (URI checkpoint) policies. Caches per descriptor; provides `get(slot_config, policy_env_info, device) -> Policy`. Ensures `initialize_to_environment` is called.
- **Slot masks**: Define per-env-agent boolean masks keyed by slot name; stored in `ComponentContext` for rollout/losses. Default builds one mask covering all agents for the lone student slot.
- **Experience spec extensions**: Add `slot_name` (int/string index) and `is_trainable_agent` flags so losses can filter minibatches without legacy PPO. Add optional `loss_profile_id` for slots so sliced losses can select the correct subset without bespoke masks.
- **Env metadata**: Extend or companion to `PolicyEnvInterface` to expose `num_agents` and allow a mapping `agent_idx -> slot_name`. Keep old code paths working by defaulting to single-student mapping. Ensure env/recipe can set per-env-agent slot assignment (enables slot-based kickstarting: some agents use teacher-supervised loss, others pure PPO).
- **Backward compatibility**: If `agent_slots` absent, auto-create a single `student` slot using the existing policy. Losses continue to work. If `loss_profile` not set, default to PPO-compatible profile.
- **Docs**: Brief README snippet and sample config showing two slots (student + scripted teacher).

### Acceptance criteria
- Running training with no new config behaves exactly as today.
- Enabling `agent_slots` with two slots (one trainable, one scripted) completes rollout; TD carries `slot_name`, `loss_profile_id`, and `is_trainable_agent`; replay stores them.
- No reliance on legacy monolithic PPO.

### Slot-aware losses groundwork
- Define `LossProfile` config mapping a name → set of losses to run for agents in that profile (e.g., `ppo_only`, `kickstarter_supervised`). Phase 0 only introduces config wiring and tagging in TD; actual selective execution can land in Phase 1 but the data must be present now.
- Teach sliced/kickstarter paths to read `loss_profile_id` when present, falling back to masks if absent. This keeps #4107 behavior viable when we switch to slots.

## Phase 1 – Training Pipeline (summary)
- Rollout forwards each slot policy, merges actions by masks, sets metadata for losses.
- Loss specs include slot/trainable flags; sliced losses AND their masks with trainable mask.

## Phase 2 – Evaluation Pipeline (summary)
- Eval config accepts slots; runner loads per-slot checkpoints/scripted classes; metrics per slot.

## Phase 3 – Dual-Policy Details (summary)
- Frozen NPC NN or co-training; masks decide gradient flow; optional alternating optim steps.

## Phase 4 – Scripted/Cogames Integration (summary)
- Scripted policies load via `class_path` through registry; mettagrid Puffer teachers map to a slot.

## Phase 5 – Backward Compatibility & Migration (summary)
- Auto-expand legacy knobs; deprecate legacy PPO; feature-flag loss masking initially.

## Phase 6 – Testing (summary)
- Unit (registry, masks), integration (dual rollout), E2E recipe with eval vs scripted opponent.
