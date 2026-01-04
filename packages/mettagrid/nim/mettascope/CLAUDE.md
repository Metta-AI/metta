# MettaScope - AI Assistant Guide

MettaScope is a visualization and debugging tool for MettaGrid simulations. It renders replays, allows real-time
interaction with simulations, and provides debugging panels for agents and objects.

## Architecture Overview

MettaScope is written in Nim and compiles to:
1. **Standalone binary** (`mettascope.out`) - for viewing replay files directly
2. **Dynamic library** (`libmettascope.dylib/.so/.dll`) - embedded in Python via bindings for real-time rendering

```
┌─────────────────────────────────────────────────────────────────┐
│                         Python Layer                            │
│  (mettagrid/renderer/mettascope.py - loads libmettascope)       │
└─────────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────────┐
│                    bindings/bindings.nim                        │
│  Exports: init(), render() - Python/Nim interface via genny    │
└─────────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────────┐
│                      src/mettascope.nim                         │
│  Main entry point, window management, panel orchestration       │
└─────────────────────────────────────────────────────────────────┘
                              │
        ┌─────────────────────┼─────────────────────┐
        ▼                     ▼                     ▼
┌───────────────┐   ┌─────────────────┐   ┌─────────────────┐
│  worldmap.nim │   │   panels.nim    │   │   replays.nim   │
│  Grid render  │   │   UI layout     │   │  Data loading   │
└───────────────┘   └─────────────────┘   └─────────────────┘
```

## Key Modules

| Module | Purpose |
|--------|---------|
| `mettascope.nim` | Main entry, window setup, frame loop, atlas generation |
| `replays.nim` | Load/parse replay files, Entity/Replay data structures |
| `worldmap.nim` | Render grid, agents, objects, terrain tiles |
| `panels.nim` | Dockable panel system, layout management |
| `objectinfo.nim` | Object/agent info panel content |
| `common.nim` | Shared state: window, replay, step, selection |
| `vibes.nim` | Vibe selector panel |
| `timeline.nim` | Playback scrubber |
| `pathfinding.nim` | A* pathfinding for click-to-move |
| `tilemap.nim` | GPU-based tilemap rendering |
| `bindings.nim` | Python FFI via genny library |

## Data Flow

### Replay Loading (Historical Mode)
```
JSON.z file → zlib decompress → JSON parse → replays.nim:loadReplay()
                                                    │
                                                    ▼
                                           Replay object with:
                                           - objects: seq[Entity]
                                           - config: Config
                                           - actionNames, itemNames, etc.
```

### Real-time Mode (Python Integration)
```
Python simulation → JSON string → bindings:render() → replay.apply()
                                                           │
                                                           ▼
                                                  Updates Entity time series
```

### Rendering Pipeline
```
onFrame() → playControls() → sk.beginUI()
                │
                ├─► drawHeader()
                ├─► drawTimeline()
                ├─► drawFooter()
                └─► drawPanels()
                        │
                        └─► drawWorldMap() → worldmap.nim:drawWorldMap()
                                                │
                                                ├─► drawTerrainMap()
                                                ├─► drawObjects()
                                                ├─► drawAgents()
                                                └─► drawSelectionHighlight()
```

## Building

### Prerequisites
- Nim 2.2.4+ (`brew install nim` or from nim-lang.org)
- nimby package manager (`nimble install nimby`)

### Build Commands

```bash
cd packages/mettagrid/nim/mettascope

# Sync dependencies
nimby sync -g nimby.lock

# Build standalone binary (also regenerates atlas)
nim c -d:release src/mettascope.nim

# Build Python bindings library
nimble bindings
```

### Full Rebuild via Package Install
```bash
cd /path/to/metta
METTAGRID_FORCE_NIM_BUILD=1 uv pip install -e packages/mettagrid
```

## Asset System

### Silky Atlas
MettaScope uses a texture atlas system (`silky`) for efficient rendering. The atlas is generated at **runtime** when
the standalone binary starts (in the `when isMainModule` block).

**Atlas sources** (from `data/` folder):
- `theme/` - UI elements, 9-patch panels
- `ui/` - Icons, buttons
- `vibe/` - Emoji icons for vibes (32x32 PNGs)
- `resources/` - Resource icons

**Generated files:**
- `data/silky.atlas.png` - Combined texture
- `data/silky.atlas.json` - Sprite coordinates

### Adding New Vibes

1. Add PNG to `data/vibe/{name}.png` (32x32, from noto-emoji)
2. Rebuild atlas by running standalone binary:
   ```bash
   cd /path/to/metta
   packages/mettagrid/nim/mettascope/src/mettascope.out
   # (will crash looking for replay, but atlas is regenerated first)
   ```
3. Rebuild bindings:
   ```bash
   METTAGRID_FORCE_NIM_BUILD=1 uv pip install -e packages/mettagrid
   ```

**Emoji to PNG mapping:**
```
🌀 scrambler → emoji_u1f300.png → data/vibe/scrambler.png
🔗 aligner   → emoji_u1f517.png → data/vibe/aligner.png
```

Get PNGs from: `https://github.com/googlefonts/noto-emoji/tree/main/png/32/`

## Debugging

### Common Issues

**1. Vibe/icon not rendering**
- Check if PNG exists in `data/vibe/{name}.png`
- Verify it's in `data/silky.atlas.json` (search for `"vibe/{name}"`)
- Rebuild atlas (see above)

**2. Replay not loading**
- Check JSON structure matches `docs/replay_spec.md`
- Verify zlib compression: `python -c "import zlib; print(zlib.decompress(open('file.json.z','rb').read()))"`
- Check `type_names` array includes all object types

**3. Black screen / rendering issues**
- Ensure OpenGL context is valid
- Check `worldMapZoomInfo` initialization
- Verify atlas files exist and are readable

**4. Python bindings crash**
- Check `bindings/generated/` has `.dylib` and `.py` files
- Verify Nim version matches (2.2.4)
- Look for stack trace in terminal output

### Debug Logging

Add echo statements in Nim code:
```nim
echo "Debug: step=", step, " selection=", selection.id
```

Errors are caught and logged in `onFrame()`:
```nim
try:
  drawWorldMap(worldMapZoomInfo)
except:
  echo "Error in drawWorldMap: ", getCurrentExceptionMsg()
```

### Inspecting Replay Data

```nim
# In any module with access to common:
echo "Objects: ", replay.objects.len
echo "Max steps: ", replay.maxSteps
for obj in replay.objects:
  if obj.isAgent:
    echo "Agent ", obj.agentId, " at ", obj.location.at(step)
```

## Key Data Structures

### Entity (replays.nim)
```nim
Entity* = ref object
  id*: int                      # Unique object ID
  typeName*: string             # "agent", "wall", "assembler", etc.
  location*: seq[IVec2]         # Time series of positions
  orientation*: seq[int]        # Time series of rotations
  inventory*: seq[seq[ItemAmount]]
  vibeId*: seq[int]             # Time series of vibe IDs
  # Agent-specific:
  agentId*: int
  actionId*: seq[int]
  totalReward*: seq[float]
  # ... more fields
```

### Time Series Access
```nim
# Get value at current step:
let pos = entity.location.at(step)
let vibe = entity.vibeId.at(step)

# The .at() proc handles:
# - Empty sequences (returns default)
# - Out of bounds (clamps to last value)
```

### Common State (common.nim)
```nim
var
  window*: Window               # windy window
  replay*: Replay               # Current replay data
  step*: int                    # Current simulation step
  selection*: Entity            # Currently selected object
  play*: bool                   # Is playback running
  playSpeed*: float32           # Playback speed multiplier
  playMode*: PlayMode           # Historical or Realtime
```

## Real-time Integration

The Python renderer calls:
1. `init(dataDir, replayJson, autostart)` - Initialize window and load initial state
2. `render(step, replayStepJson)` - Update state and render frame

The `render()` function:
- Calls `replay.apply(replayStepJson)` to update entity time series
- Runs the frame loop until `requestPython` is set
- Returns action requests from user clicks

## Performance Notes

- Tilemap uses GPU instancing for terrain
- Atlas packing minimizes texture switches
- Time series are expanded on load (not binary searched)
- Panel layout is cached until window resize

## Testing

```bash
cd packages/mettagrid/nim/mettascope
nim c -r tests/test_replays.nim
nim c -r tests/test_pathfinding.nim
```


