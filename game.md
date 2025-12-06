# MettaGrid Game Notes

## Entities and Map
- Grid stores exactly one `GridObject` per cell; overlaps are rejected on add/move.
- Static: `Wall` only. Walls never co‑locate with anything else.
- Dynamic: `Agent`, `Assembler` (also used for charger/extractors), `Chest`.
- No static+dynamic sharing; a cell is empty | wall | one dynamic object.

## Actions and Step Order
- Actions: `noop`, `move`, `change_vibe`, `attack` (attack currently has no target lookup and effectively fails).
- Per step (`MettaGrid::_step`):
  1) Zero rewards/obs/action_success.
  2) Increment `current_step`; shuffle agents.
  3) For each priority (attack=1, others=0) run actions across agents; move consumes required resources after success and refuses occupied targets unless `Usable::onUse` triggers.
  4) Inventory regen tick if interval divides step.
  5) Clipper may clip assemblers (if enabled).
  6) Compute observations (see below).
  7) Compute stat-based rewards; accumulate episode rewards; set truncations/terminals if `max_steps` hit.

## Station Behavior (CvC missions)
- Assemblers (heart/gear crafting) consume inputs from adjacent agents based on vibes/count, produce outputs to them; track cooldown/max uses; can be clipped/unclipped.
- Chargers/Extractors are assemblers with output-only protocols (energy/resources), sometimes partial-use during cooldown.
- Chests transfer resources based on actor vibe (deposit/withdraw presets for carbon/oxygen/germanium/silicon/heart).
- All are stationary dynamic objects.

## Observations (capacity 200 tokens in CvC)
- Window must fit packed coords (≤15×15).
- Per agent per step (`_compute_observation`):
  - Emit globals: episode completion %, last action, last reward, optional goals, optional compass (toward map center).
  - Iterate `ObservationPattern` (Manhattan shells) over window; pack location byte per cell.
  - Non-optimized path: fetch `Grid::object_at`; call `obs_features()` and append tokens until capacity.
  - Optimized path: use per-cell cache:
    - Static cache built once from walls.
    - Dynamic cache rebuilt each step (today) from all non-wall objects.
    - Copy cached tokens with current location; order: globals → cells in pattern order → static then dynamic.
  - Stats track tokens_written/dropped/free_space.

## Caching Implications
- Only walls are static; dynamic objects never share a cell with walls or each other, so each cell has at most one token source.
- This allows a fixed-capacity slab per cell (e.g., max 8 tokens) without needing static/dynamic interleaving rules.
