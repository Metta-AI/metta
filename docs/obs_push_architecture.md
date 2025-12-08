# Observation Update: Simple Push Setup

Based on the current `mettagrid_c.cpp/.hpp` (full rebuild every step) and the prior dirty-bit experiment now parked in
`dirtybit_mettagrid_c.cpp/.hpp`, we can drop dirty-bit read/write entirely and add a minimal push-style updater.

## Baseline + Backup

- Keep `mettagrid_c.cpp/.hpp` on the clean origin/main implementation (clears obs in `_step` and calls
  `_compute_observations`).
- Preserve the dirty-bit version in `dirtybit_mettagrid_c.cpp/.hpp` for reference and benchmarking (it precomputes
  `_obs_pattern`, caches per-cell tokens, tracks assembler adjacency, and logs per-cell truncation).
- No new dirty flags or per-cell caches in the main path; fall back to full rebuild on reset or if a push path ever
  desyncs.

## `updateObservation` Helper

- Input: a cell location (and optional object pointer/null). Compute fresh tokens for that cell via
  `_obs_encoder->encode_tokens` into a small stack buffer; if empty, use an empty token set.
- For each agent: if `abs(dr) <= obs_height/2` and `abs(dc) <= obs_width/2`, compute the packed location
  (`PackedCoordinate::pack(dr + obs_height/2, dc + obs_width/2)`), scan that agent’s obs buffer for tokens matching that
  location, and rewrite them in-place (drop extras with `EmptyTokenByte`, extend when fewer existed).
- No per-cell offset bookkeeping; the packed location already lives in each token (see `mettagrid_c.cpp`’s
  `_compute_observation` loop over `PackedCoordinate::ObservationPattern`).
- Keep a tiny scratch buffer and reuse it to avoid allocations; keep the 24-token-per-cell truncation log pattern from
  `dirtybit_mettagrid_c.cpp` if helpful.

## Movement Slice

- Add `shiftObservation(agent_idx, dr, dc)` that walks an agent’s tokens, updates their packed locations, and drops any
  that move outside the window.
- After a successful move, call `shiftObservation` and then `updateObservation` on the fringe cells that became newly
  visible (one row/col depending on N/S/E/W). If the agent started/ended near a boundary, fall back to
  `_compute_observation` for that agent only.

## Where to Call `updateObservation`

- Move success: shift the mover’s window, then `updateObservation` on the source cell, destination cell, and any swapped
  occupant.
- Action handlers that mutate objects: `Attack` (target cell, attacker cell if stats/inventory change), `ChangeVibe`,
  any inventory transfers, freeze/unfreeze, vibe changes, or stat deltas that affect `obs_features`.
- Global/system ticks: inventory regeneration, assembler cooldown/use, clipper start/unclip toggles, vibe transfers in
  chests, protocol switches, and any spawn/despawn.
- Grid-level spawns/destructions (agent death, object removal/addition) should `updateObservation` the affected cell(s).

## Landing Steps

1. Copy current `mettagrid_c.cpp/.hpp` to `dirtybit_mettagrid_c.cpp/.hpp` (done) so the dirty-bit path stays available.
2. Reset `mettagrid_c.*` to origin/main to remove dirty-bit read/write.
3. Add `updateObservation` and `shiftObservation` helpers that operate directly on the existing observation buffer
   layout (no extra caches).
4. Wire calls into the mutation sites above; keep `_compute_observations` for reset and as a safety fallback.
5. Reuse the token-cap/truncation logging pattern if we exceed per-cell capacity while patching.
