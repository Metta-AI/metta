# Slot-based Multi-Policy Training and Evaluation

## Key concepts
- **policy_slots**: list of policies (checkpoint URI or class path) with metadata (trainable flag, loss_profile, device).
- **agent_slot_map**: length = num_agents; assigns each agent index to a slot id.
- **loss_profiles**: named sets of losses; losses run only on rows whose profile matches and are trainable when applicable.
- **slot_id / loss_profile_id / is_trainable_agent**: per-row metadata injected during rollout and stored in replay.

## Training config sketch
```yaml
trainer:
  policy_slots:
    - id: main
      use_trainer_policy: true
      trainable: true
      loss_profile: ppo_only
    - id: scripted_teacher
      class_path: tests.rl.fake_scripted_policy.FakeScriptedAgentPolicy
      policy_kwargs: {multiplier: 10}
      trainable: false
      loss_profile: teacher_only
  agent_slot_map: [scripted_teacher, scripted_teacher, main, main]
  loss_profiles:
    ppo_only: {losses: [ppo_actor, ppo_critic]}
    teacher_only: {losses: []}
  losses:
    ppo_actor: {profiles: [ppo_only]}
    ppo_critic: {profiles: [ppo_only]}
    ppo: {enabled: false}
```

## Evaluation config sketch
```yaml
simulation:
  policy_slots:
    - id: student
      policy_uri: s3://my/checkpoint
    - id: scripted_teacher
      class_path: tests.rl.fake_scripted_policy.FakeScriptedAgentPolicy
      policy_kwargs: {multiplier: 10}
  agent_slot_map: [scripted_teacher, student]
```
- Runner loads slots via `SlotRegistry`, assigns per-agent control via `agent_slot_map`, and reports per-slot returns/winrate.

## Defaults
- If no `policy_slots` provided: a single `main` slot uses the trainer policy; map covers all agents.
- Loss profiles optional; if omitted, losses run on all rows (respecting `is_trainable_agent` for trainable-only losses).

## Notes
- Trainable flag freezes non-trainable slots (`requires_grad=False`).
- TD merge writes only action/logprob/value keys from each slot to avoid clobbering metadata.
```
