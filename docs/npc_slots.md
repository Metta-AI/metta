# NPC slots with SimpleNPCPolicy

- Use the canonical NPC baseline via a slot entry: `class_path: metta.agent.policies.npc.SimpleNPCPolicy`, set
  `trainable: false`, and give it a distinct `loss_profile` if you want to exclude PPO losses.
- Map agents to the NPC slot with `agent_slot_map` (length = num_agents). Example: `["npc", "student"]` assigns agent 0
  to NPC and agent 1 to the student policy.
- For eval-only scenarios, point `policy_uri` at your student checkpoint and keep the NPC slot as scripted. See
  `recipes/examples/slots_npc_eval.yaml`.
- `SimpleNPCPolicy` emits valid `actions`, `act_log_prob`, `entropy`, and `values`, so it plugs cleanly into
  `SlotControllerPolicy`, `SlotRegistry`, and losses expecting those keys.
- For training with a frozen NPC slot and a learnable student, see `recipes/examples/slots_student_npc_train.yaml`
  (student PPO losses run on the `ppo_only` profile; NPC rows are excluded).
