# MettaGrid Observation Tokens

This README documents the raw observation layout emitted by the MettaGrid simulator. Each agent receives its egocentric view as a fixed-size array of 3-byte tokens (`location`, `feature_id`, `value`).

## Observation Tensor Layout

- **Shape:** `(num_tokens, 3)` per agent. Defaults are an 11×11 window, `token_dim = 3`, `num_tokens = 200` (`ObsConfig`).
- **Encoding:** Column `0` stores a packed coordinate, `1` the feature id, `2` the feature value.
- **Packing:** Coordinates use four bits each (`row << 4 | col`). `0xFF` means “empty slot”. Rows/cols must be ≤ 14 (`PackedCoordinate`).
- **Ordering:** `_compute_observations` writes tokens in three passes:
  1. **Global/agent-local features** at the center cell.
  2. Optional **compass** tile (one step toward assembler hub).
  3. Grid objects in increasing Manhattan distance from the agent. Tokens beyond `num_tokens` are dropped.
- **Agent origin:** The observing agent is placed at `(obs_height // 2, obs_width // 2)` when encoding.

## Feature Catalogue

`IdMap._compute_features()` assigns stable ids and normalization factors (available via `Simulation.id_map.features()`). Every map configuration includes:

- **Core agent flags** – `agent:group`, `agent:frozen`, `agent:orientation`, `agent:reserved_for_future_use`, `converting`, `swappable`.
- **Global scalars** – `episode_completion_pct`, `last_action`, `last_action_arg`, `last_reward`.
- **Agent extras** – `vibe`, `agent:visitation_counts`, `agent:compass`, `tag`.
- **Object state** – `cooldown_remaining`, `clipped`, `remaining_uses`.
- **Inventory tokens** – One `inv:<resource>` per resource. When protocol-detail observations are enabled, `protocol_input:<resource>` and `protocol_output:<resource>` are also generated.
- **Tags:** Use `IdMap.tag_names()` to decode tag ids to strings.

Feature values are clipped to `uint8`. Use the `normalization` field in `ObservationFeatureSpec` to rescale if needed.

## Token Sources

| Source | Location | Feature ids | Values |
| ------ | -------- | ----------- | ------ |
| **Global block** | Agent center | Configured subset of `agent:*`, `inv:*`, and global flags. | Group id, frozen flag, vibe id, resource counts, visit counts, last action/reward, etc. |
| **Compass hint** | Neighbor cell toward assembler (if enabled) | `agent:compass` | Constant sentinel `1`; absence means no hint. |
| **Agents** | Actual agent positions (including self) | `agent:group`, `agent:frozen`, `vibe`, `inv:*`, `tag`. | Group id, frozen flag, vibe id, inventory counts, tag ids. |
| **Walls** | Wall tiles | `swappable` (if true), `tag`, `vibe`. | Binary swappable flag, tag ids, vibe id. |
| **Assemblers / chargers / extractors** | Building tile | `cooldown_remaining`, `clipped`, `remaining_uses`, `tag`, `vibe`, plus `protocol_input:/output:` fields when protocol details are on. | Cooldown ticks, clipped flag, remaining uses (≤255), current recipe inputs/outputs, tag ids, vibe. |
| **Chests** | Chest tile | `vibe`, `inv:*`, `tag`. | Stored resource amounts (≤255), tag ids. |
| **Other objects** | Their grid location | Whatever `GridObject::obs_features()` returns. | Object-specific payload.

### Additional Notes

- Visitation counts emit up to five tokens (center, up, down, left, right) when `global_obs.visitation_counts` is enabled.
- `last_reward` is scaled by 100 before casting to `uint8`. Similar scaling applies to other features—`normalization` tells you the intended range.
- Protocol inputs/outputs appear only when `GameConfig.protocol_details_obs` is true.
- Tokens beyond the configured limit are dropped; statistics for dropped tokens are tracked internally (`tokens_dropped`).

## Decoding Tokens

1. **Unpack coordinates:** Use `PackedCoordinate.unpack(token.location)` to recover `(row, col)`.
2. **Lookup feature metadata:** `Simulation.id_map.feature(token.feature_id)` returns an `ObservationFeatureSpec` with name and normalization.
3. **Interpret the value:** Scale or interpret `token.value` using the feature name and `normalization` metadata (e.g., divide by `normalization` to get a 0–1 signal).

This README should help you decode the raw observation tensor for logging, debugging, or feeding model inputs.
