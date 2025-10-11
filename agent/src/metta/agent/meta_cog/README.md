# MetaCog (MC) Actions: Design, Flow, and Implementation Guide

This document explains how MetaCog (MC) actions work in this repository: what they are, how they
are wired into the policy two-pass loop, how they should be implemented in components, and how to
extend the system. It is written for engineers and LLM agents to gain full understanding to build
and modify MC-enabled components.

## What are MC actions?

MC actions are internal, per-environment control signals that the policy can choose at every step
to mutate transient state inside the network before running the main forward pass. Examples:

- Shrink a token observation window for a specific environment (focus).
- Clear (zero-out) an LSTM’s hidden/cell state for selected environments.
- Inject Gaussian noise into an LSTM’s hidden/cell state to encourage exploration or robustness.

Key properties:

- Internal only: They do not affect the environment directly; they mutate model-internal buffers
  or masks that influence the subsequent forward pass.
- Per-env granularity: Actions are indexed by `env_id` so two sets of workers (double buffering)
  can be handled uniformly.
- Transient: Many MC effects are single-step and should be reset automatically after use.

## The two-pass policy loop

At each environment step:

1) MC-action selection pass
   - The policy produces logits/probabilities over internal actions (across all components).
   - The selection (e.g., sampling or argmax) yields per-env chosen MC actions.
   - The framework calls `apply_mc_actions`, which dispatches env_ids to the `MetaCogAction`
     handlers attached to components.

2) Main pass
   - Components have now mutated their transient internal state (e.g., focus masks, pending
     per-step LSTM controls) and the policy runs its main forward pass again over the original
     observations.
   - After this pass, components may reset their single-step state to prepare for the next step.

This loop ensures the policy can “think first” (choose control), then “act” with the altered
network state.

## Core types and files

- `mc.py` – Core MC plumbing and base definitions (e.g., `MetaCogAction`).
- `mc_policy_auto_builder.py` – Gathers MC actions from components, assigns indices, and applies
  the selected actions per step.
- Components implementing actions:
  - `mc_obs_shim_tokens.py` – Token observation shaping and per-env focus windows.
  - `mc_lstm_reset.py` – LSTM state resets, per-step noise injection, and clear actions.
  - `mc_vit_reset.py` – Example Vision Transformer-related resets (if present).

## Registering actions in components

Inside a component, define actions as `MetaCogAction` members and attach their apply methods:

```python
self.noise_1 = MetaCogAction("lstm_noise_1")
self.noise_1.attach_apply_method(self.inject_noise_1)
```

The `mc_policy_auto_builder` will discover these actions, allocate integer IDs, and call the
attached method with a 1D `env_ids` LongTensor when that action is selected.

Implementation patterns:

- Prefer vectorized indexing over Python loops: `buffer[env_ids, :] = ...`.
- Register per-env state as buffers so they move with `.to(device)`. Use `persistent=False` for
  ephemeral caches/masks so checkpoints stay small and free of stale content.
- Support dynamic capacity growth when new `env_ids` appear.

## Rollout vs Training behavior

The batch shape differs between rollout and training:

- Rollout: [B, ...] (TT == 1)
- Training: [B*TT, ...] (unrolled through time)

Guidelines:

- Always derive `env_ids` from `training_env_ids` and reshape to [B*TT] so components see a
  consistent indexing view.
- For single-step, per-env effects (e.g., token focus), mutate a per-env buffer and reset it after
  the forward pass. Example: `mc_obs_shim_tokens.py` resets focus masks for envs that used them.
- For LSTM controls during training, schedule per-row instructions and apply them per timestep in
  the unrolled loop. Example: `mc_lstm_reset.py` stores `pending_noise_code`/`pending_clear_rows`
  shaped [B*TT], reshapes to [B, TT], applies them each step in `LstmTrainStep.forward`, and then
  zeros the pending buffers after use.

## Example: Per-env token focus (mc_obs_shim_tokens.py)

Design:

- `focus_mask`: [num_envs, default_max_tokens] boolean buffer. True = keep; False = mask.
- Two MC actions: `focus_1`, `focus_2` that set a temporary window per env.
- After global crop to length L (max dense length clamped by default_max_tokens), the per-env mask
  is intersected to narrow the view further.
- After the pass, masks for envs that had focus applied are reset back to all-True.

Notes:

- `training_env_ids` must be shaped [B*TT]. If missing, synthesize a contiguous range.
- Buffers use `persistent=False` to avoid checkpointing ephemeral state. They still move with the
  module across devices.

## Example: Per-step LSTM clear and noise (mc_lstm_reset.py)

Rollout (TT == 1):

- `clear(env_ids)` zeros `lstm_h/lstm_c` for those envs immediately.
- `inject_noise_1/2(env_ids)` adds sampled Gaussian noise via `torch.normal` directly to buffers.

Training (TT > 1):

- Scheduling: `inject_noise_1/2` mark `pending_noise_code[rows]` (1 → std 0.1, 2 → std 0.5), and
  `clear` marks `pending_clear_rows[rows]`. Here `rows = B*TT`.
- Application per step:
  - Before unrolling, reshape to [B, TT] and map codes to `noise_std_bt` with per-env std values.
  - In `LstmTrainStep.forward`, at timestep t:
    - Clear: OR the clear row into the reset mask and zero `h_t/c_t`.
    - Noise: `std_t = noise_std_bt[:, t]` broadcast to [1, B, 1]; add
      `noise = torch.randn_like(h_t) * std_broadcast` to `h_t/c_t`.
  - After unrolling, zero out the pending buffers.

No-op safety:

- If no MC actions are called for a batch, pending buffers remain zero/False and have no effect.
- In rollout, if no MC action is called, buffers remain unchanged.

## Device handling and persistence

- All per-env state is registered as buffers so `.to(device)` moves them automatically.
- Use `persistent=False` for transient/control buffers (e.g., focus masks, pending schedules). They
  are reconstructed in `__init__` and grown when needed, avoiding checkpoint issues.

## Extension checklist

1. Define `MetaCogAction` members on your component and `attach_apply_method` for each.
2. Decide if your action is single-step transient (like focus) or must be applied per timestep
   in training (like LSTM noise/clear). Choose buffer strategy accordingly.
3. Use vectorized indexing for per-env updates. Avoid loops.
4. Support dynamic capacity growth when new env ids appear.
5. Add validation and clamping as needed (e.g., window sizes below a maximum).
6. Document the behavior, especially rollout vs training differences and reset semantics.

## Quick reference: common shapes

- `training_env_ids`: [B*TT]
- Rollout: TT == 1, rows == B
- Training: TT > 1, rows == B*TT; unroll in `LstmTrainStep` over t in [0, TT)
- Per-env buffers (focus, persistent LSTM): indexable by env_id
- Per-row pending buffers (training-only): [B*TT] then reshaped to [B, TT]

## FAQ

Q: Why not store transient masks in checkpoints?

A: They are single-step control state, often sized by the current batch or env count. Restoring
them from checkpoints can cause shape/device mismatches and stale semantics. Recreating and growing
them on demand is robust and cheap.

Q: How do MC actions relate to external actions sent to the environment?

A: They are independent. MC actions control the network’s internal computation before the main
forward pass. External actions are emitted after the main pass and sent to the environment.

Q: How do I add a new MC action that needs per-timestep training logic?

A: Follow `mc_lstm_reset.py`: create a pending [B*TT] buffer (non-persistent), fill it from the
action handler during training, reshape to [B, TT] inside `forward`, and apply it inside the
unrolled loop.


