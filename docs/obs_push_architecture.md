# Observation Update: Push-Based Architecture (No Dirty Bits)

This describes a push-driven observation updater that keeps per-agent buffers accurate without sweeping dirty bits. Moves are fast (shift the window), and world changes push token updates directly into the agents that can see them.

## Core Data Structures
- **obs_buffers[agent]**: current observation buffer for each agent (tokens with packed locations).
- **cell_tokens[cell]**: current token list for each cell (static+dynamic). Refresh only when that cell’s object changes.
- **fov_map[agent]**: precomputed list of cell indices in the agent’s FoV (using the packed offset pattern). Also build a reverse index: **cell -> list of agents** who can see it.
- **cell_offsets[agent][cell]**: for cells currently in that agent’s FoV, the start/end positions in the agent’s obs buffer. Updated when windows shift or when inserts/removals happen.
- **assembler_cells**: list of assembler (and other neighbor-dependent) cell indices for quick adjacency checks.

## Step Order (matches current pipeline: actions → regen → clipper → obs)
1) **Shift from prior obs**  
   - Copy prior obs buffers to “next”.  
   - For each agent that successfully moved N/S/E/W, shift its window: rewrite packed locations and adjust `cell_offsets`. No clearing.

2) **Collect change events as the world mutates**  
   - Move success: record `move_event(src_cell, dst_cell, object_id)`.  
   - Inventory/vibe changes: `content_event(cell)`.  
   - Time-based ticks: emit `cooldown_event(cell)` per assembler (or only those with active cooldown).  
   - Clipper/start-clipped: after clipper init/unclip, emit `content_event(cell)` for affected assemblers.

3) **Apply change events (push updates)**  
   For each event cell:
   - Recompute `cell_tokens[cell]` if needed.  
   - For each agent in `cell -> agents`:
     - If the cell is already in view (`cell_offsets` exists): overwrite that span with the new tokens (truncate/log if too many, pad tail if fewer).  
     - If the cell just entered view due to the agent’s shift: insert the new tokens at the proper position, shift the tail, and update `cell_offsets` for subsequent cells.  
     - If the cell left view: remove its span and compact (or overwrite with next cell’s tokens and pad tail).

4) **Neighbor-dependent objects (assemblers, etc.)**  
   - For each `move_event`, if `src` or `dst` is adjacent to an `assembler_cell`, emit `content_event(assembler_cell)`. This keeps recipe/synergy tokens current without dirtying every move.

5) **Finalize**  
   - Update `cell_offsets` for agents whose windows shifted or whose windows had inserts/removals.  
   - No global recompute; no per-agent rebuild unless a change touched their FoV.

## Precomputation & Token Rules
- Precompute `fov_map` and the reverse map at reset; reuse each step.  
- Keep per-cell token order stable: static tokens first, then dynamic, so overwrites are predictable.  
- Keep the per-cell cap at 24; log on truncation to avoid silent drops.  
- Construct the clipper before the first obs; run start-clipped/unclip, then push `content_event` to those cells so reset observations are accurate.

## Why This Works
- **Performance**: Moves stay O(window_size) via shift; updates are O(changed_cells × agents_in_view × token_count) with a cheap reverse map and few assembler updates.  
- **Correctness**: Cooldowns and neighbor changes are pushed directly; no stale cached cells.  
- **Simplicity**: No dirty sweeps; events drive the minimal writes needed to keep observations correct.
