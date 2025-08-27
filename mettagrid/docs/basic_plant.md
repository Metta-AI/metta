### basic_plant – full implementation guide (and lessons learned)

This object is a wall-like building that replicates straight north every N ticks.

- Name: `basic_plant`
- ASCII: `x`
- Behavior: every `grow_ticks`, spawns a copy of itself at (r-1, c) on the object layer if empty
- Stops when blocked or at the top border

#### Where it’s implemented

- C++ object and event: `objects/plant.hpp`, `objects/constants.hpp` (adds `PlantGrow`)
- Engine registration and Python bindings: `mettagrid_c.cpp`
- Python config model: `mettagrid/mettagrid_config.py` (`PlantConfig`)
- Python → C++ config conversion: `mettagrid/mettagrid_c_config.py`
- Default object registration for arena envs: `mettagrid/config/envs.py`
- ASCII encoders: `mettagrid/char_encoder.py` and `gridworks/src/lib/encoding.json`

### 1) C++ engine changes

1.1 Add a new event type for growth

```10:14:mettagrid/src/metta/mettagrid/objects/constants.hpp
enum EventType {
  FinishConverting = 0,
  CoolDown = 1,
  PlantGrow = 2,
  EventTypeCount
};
```

Why:

- Growth is time-based and must occur across steps without agent actions. The engine’s `EventManager` dispatches timed
  logic. Adding `PlantGrow` lets us schedule growth at specific timesteps decoupled from the action loop.

  1.2 Implement PlantConfig, Plant, and PlantGrowHandler

Key points:

- Call `GridObject::init(...)` to initialize type and location.
- Only schedule events after the object is on the grid (id != 0). We do this in `set_event_manager` which is called
  after `add_object`.
- Use `grid->add_object(child)` to create new plants; this assigns an id and occupies the cell.
- Place only if the target object layer is empty.

```12:87:mettagrid/src/metta/mettagrid/objects/plant.hpp
struct PlantConfig : public GridObjectConfig {
  PlantConfig(TypeId type_id, const std::string& type_name, unsigned int grow_ticks)
      : GridObjectConfig(type_id, type_name), grow_ticks(grow_ticks) {}
  unsigned int grow_ticks;
};

class Plant : public GridObject {
public:
  unsigned int grow_ticks;
  EventManager* event_manager;

  Plant(GridCoord r, GridCoord c, const PlantConfig& cfg)
      : grow_ticks(cfg.grow_ticks), event_manager(nullptr) {
    GridObject::init(cfg.type_id, cfg.type_name, GridLocation(r, c, GridLayer::ObjectLayer));
  }

  void set_event_manager(EventManager* em) {
    this->event_manager = em;
    if (this->event_manager && this->id != 0) {
      this->event_manager->schedule_event(EventType::PlantGrow, grow_ticks, this->id, 0);
    }
  }

  void grow_once() {
    if (this->event_manager && this->id != 0) {
      this->event_manager->schedule_event(EventType::PlantGrow, grow_ticks, this->id, 0);
    }
    Grid* grid = this->event_manager ? this->event_manager->grid : nullptr;
    if (!grid || this->location.r == 0) return;
    GridCoord nr = static_cast<GridCoord>(this->location.r - 1);
    GridCoord nc = this->location.c;
    if (!grid->is_empty_at_layer(nr, nc, GridLayer::ObjectLayer)) return;
    Plant* child = new Plant(nr, nc, PlantConfig(this->type_id, this->type_name, this->grow_ticks));
    if (grid->add_object(child)) child->set_event_manager(this->event_manager);
    else delete child;
  }

  std::vector<PartialObservationToken> obs_features() const override {
    std::vector<PartialObservationToken> features;
    features.push_back({ObservationFeature::TypeId, static_cast<ObservationType>(this->type_id)});
    return features;
  }
};

class PlantGrowHandler : public EventHandler {
public:
  explicit PlantGrowHandler(EventManager* event_manager) : EventHandler(event_manager) {}
  void handle_event(GridObjectId obj_id, EventArg /*arg*/) override {
    Plant* plant = static_cast<Plant*>(this->event_manager->grid->object(obj_id));
    if (!plant) return;
    plant->grow_once();
  }
};
```

Why these choices:

- `PlantConfig` holds tunables (`grow_ticks`) separate from runtime state, matching the pattern used by `Wall`,
  `Converter`, `Box`.
- `set_event_manager` is where we schedule initial growth because `Grid::add_object` assigns the `id`.
  `EventManager::schedule_event` requires a non-zero `object_id`.
- Scheduling the next `PlantGrow` before doing the spawn keeps behavior periodic even if the current spawn fails
  (blocked or border).
- We only place on `GridLayer::ObjectLayer` and only if empty to mimic wall occupancy semantics.

  1.3 Register the event handler and instantiate Plant from map

- Register `PlantGrowHandler` with the event manager.
- During map load, detect `PlantConfig` via `dynamic_cast` and add a `Plant`.

```79:86:mettagrid/src/metta/mettagrid/mettagrid_c.cpp
_event_manager->event_handlers.insert(
    {EventType::FinishConverting, std::make_unique<ProductionHandler>(_event_manager.get())});
_event_manager->event_handlers.insert({EventType::CoolDown, std::make_unique<CoolDownHandler>(_event_manager.get())});
_event_manager->event_handlers.insert({EventType::PlantGrow, std::make_unique<PlantGrowHandler>(_event_manager.get())});
```

```200:214:mettagrid/src/metta/mettagrid/mettagrid_c.cpp
const PlantConfig* plant_config = dynamic_cast<const PlantConfig*>(object_cfg);
if (plant_config) {
  Plant* plant = new Plant(r, c, *plant_config);
  _grid->add_object(plant);
  plant->set_event_manager(_event_manager.get());
  _stats->incr("objects." + cell);
  continue;
}
```

Why:

- The engine stores object configs as `GridObjectConfig` pointers. `dynamic_cast` identifies the concrete type so the
  correct subclass (`Plant`) can be constructed. This mirrors how walls/converters/boxes are created.
- Event handlers must be registered once so the `EventManager` knows which function to call when `PlantGrow` events are
  due.

  1.4 Expose PlantConfig to Python via pybind

```1009:1017:mettagrid/src/metta/mettagrid/mettagrid_c.cpp
py::class_<PlantConfig, GridObjectConfig, std::shared_ptr<PlantConfig>>(m, "PlantConfig")
    .def(py::init<TypeId, const std::string&, unsigned int>(),
         py::arg("type_id"), py::arg("type_name"), py::arg("grow_ticks"))
    .def_readwrite("type_id", &PlantConfig::type_id)
    .def_readwrite("type_name", &PlantConfig::type_name)
    .def_readwrite("grow_ticks", &PlantConfig::grow_ticks);
```

Why:

- Python builds the `GameConfig`. Exposing `PlantConfig` enables passing plant parameters from Python into C++ without
  hardcoding them in the engine.

### 2) Python models and conversion

2.1 Add PlantConfig to Python models

```140:148:mettagrid/src/metta/mettagrid/mettagrid_config.py
class PlantConfig(Config):
    """Python plant configuration (self-replicating wall)."""
    type_id: int = Field(default=0, ge=0, le=255)
    grow_ticks: int = Field(ge=1, default=25)
```

Include `PlantConfig` in `GameConfig.objects` union so it can be supplied in configs:

```201:204:mettagrid/src/metta/mettagrid/mettagrid_config.py
objects: dict[str, ConverterConfig | WallConfig | BoxConfig | PlantConfig] = Field(default_factory=dict)
```

2.2 Convert Python PlantConfig → C++ PlantConfig

```120:129:mettagrid/src/metta/mettagrid/mettagrid_c_config.py
elif isinstance(object_config, PlantConfig):
    cpp_plant_config = CppPlantConfig(
        type_id=object_config.type_id,
        type_name=object_type,
        grow_ticks=object_config.grow_ticks,
    )
    objects_cpp_params[object_type] = cpp_plant_config
```

Why:

- The Python `EnvConfig`/`GameConfig` is converted into a C++ `GameConfig` before constructing the engine. The converter
  maps Python model fields to the C++ struct, preserving type ids and tunables (e.g., `grow_ticks`).
- This central conversion point also validates and normalizes resource and action definitions for other object types;
  plants fit into the same machinery.

  2.3 Default registration (example)

Provide a default building for easy use and include it in an env preset:

```40:53:mettagrid/src/metta/mettagrid/config/building.py
basic_plant = PlantConfig(type_id=17, grow_ticks=25)
```

```24:31:mettagrid/src/metta/mettagrid/config/envs.py
objects = {
    "wall": building.wall,
    ...,
    "basic_plant": building.basic_plant,
}
```

Why:

- Having a default `building.basic_plant` makes it easy to include plants in templates and scenes without rewriting
  config.

Lesson: type_id uniqueness is required. The engine builds `object_type_names` from `cfg.objects` by `type_id`, so ensure
no collisions. These names and ids flow into renderers and replays.

### 3) ASCII/editor encodings and rendering

3.1 ASCII mappings

- We chose `x` for `basic_plant` to avoid clashing with `agent.prey` which uses `p`.

```1:32:mettagrid/src/metta/mettagrid/char_encoder.py
"basic_plant": ["x"],
```

```1:1:gridworks/src/lib/encoding.json
{"agent.agent": ["@", "A"], ..., "converter": ["c"], "basic_plant": ["x"]}
```

3.2 Map editor drawing

- The editor draws `basic_plant` using the existing `wall` tile for now.

```21:27:gridworks/src/lib/draw/Drawer.ts
  wall: [{ tile: "wall" }],
  block: [{ tile: "block" }],
  basic_plant: [{ tile: "wall" }],
  altar: [{ tile: "altar" }],
```

Why:

- Two parallel encoders exist:
  - Python-side `char_encoder.py` for ASCII map import/export in the backend (`AsciiMapBuilder` calls
    `char_to_grid_object`).
  - Frontend `gridworks/src/lib/encoding.json` for the Next.js editor and viewer utilities.
- Both must include `basic_plant` so maps can be round-tripped between ASCII and the editor.
- We avoided `p` since `agent.prey` already uses it; `x` was free and readable.

Note: Hermes renderer looks up `objects/<name>.png`. If you want a custom sprite, add `objects/basic_plant.png` to your
atlas/sprites. It’s optional for functionality; gameplay doesn’t depend on it.

### 4) Minimal replay to observe growth

4.1 Tiny ASCII map (border walls and one plant):

```1:10:mettagrid/configs/basic_plant_map.txt
########################
#......................#
#......................#
#......................#
#......................#
#..........x...........#
#......................#
#......................#
#......................#
########################
```

4.2 Runner script – generates a replay with no-op actions

```1:200:tools/run_basic_plant_replay.py
# runs an env with the ASCII map and writes a replay to outputs/replays/
```

Why this setup works:

- Growth is driven by scheduled `PlantGrow` events, not agent actions. Sending all-zero actions is sufficient to advance
  time and process events (`MettaGrid::_step` calls `_event_manager->process_events(current_step)`).
- `ReplayWriter` collects `env.grid_objects` and `object_type_names` each step; as plants replicate north, new objects
  appear in the replay stream for visualization.
- Using `AsciiMapBuilder` ensures the single `x` becomes `basic_plant` at load via `char_to_grid_object` wiring.

  4.3 Build and run

1. Build the C++ extension (macOS, Homebrew Python 3.11):
2. Generate a replay using the small ASCII map:

3. Build the C++ extension (macOS, Homebrew Python 3.11):

```bash
export PATH=/opt/homebrew/bin:$PATH
cd /Users/jacke/metta/mettagrid
python3.11 -m venv .venv
source .venv/bin/activate
pip install -U pip uv
uv pip install -e .
```

2. Generate a replay using the small ASCII map:

```bash
cd /Users/jacke/metta
source mettagrid/.venv/bin/activate
python tools/run_basic_plant_replay.py --steps 60 --out outputs/replays/basic_plant_demo.json.z
```

3. View the replay:

- Open the MettaScope UI and load the generated file `outputs/replays/basic_plant_demo.json.z` (you can drag-and-drop
  the file or use any existing replay loader tooling).

### Lessons learned / pitfalls

- Schedule events only after the object is on the grid: `Grid::add_object` assigns `id`; `EventManager::schedule_event`
  asserts `object_id != 0`.
- Always register new event handlers (e.g., `PlantGrowHandler`) with the environment’s `EventManager` during
  construction.
- Instantiate new objects during map load via `dynamic_cast` of the stored `GridObjectConfig` and `add_object(...)`.
- Keep the ASCII encodings consistent across Python (`char_encoder.py`) and the web editor
  (`gridworks/src/lib/encoding.json`). Avoid collisions (e.g., `p` used for `agent.prey`).
- For simple walls, you don’t need to override `swappable()`; default is non-swappable like `Wall`.
- `object_type_names` comes from config `objects` keyed by `type_id`; pick a free `type_id`.
- UI rendering is decoupled: editor draws from its tileset; Hermes tries `objects/<name>.png` – optional.

With the above, you can reproduce the full object pipeline: C++ engine + events → pybind config → Python models →
encoding → editor/replay.

### Wiring overview

End-to-end flow for basic_plant:

- Python config
  - `GameConfig.objects["basic_plant"] = PlantConfig(type_id=..., grow_ticks=...)`
  - Included in `GameConfig.objects` union so it serializes and validates
  - Converted by `mettagrid_c_config.convert_to_cpp_game_config` → C++ `GameConfig.objects` with `CppPlantConfig`

- Engine construction
  - `MettaGrid(GameConfig, map, seed)` builds grid and event manager
  - Registers `PlantGrow → PlantGrowHandler`
  - Parses map cells; finds config by name; `dynamic_cast<PlantConfig>` → `new Plant(r,c, cfg)` →
    `grid.add_object(plant)` → `plant.set_event_manager(em)` schedules first grow

- Step loop
  - Each step increments `current_step` and calls `event_manager.process_events(current_step)`
  - `PlantGrowHandler` calls `Plant::grow_once()` which schedules the next grow and tries `grid.add_object(child)` at
    `(r-1,c)`

- Observability & replay
  - `obs_features()` includes `TypeId` so models/UI can identify objects
  - `ReplayWriter` logs `grid_objects` and `object_type_names` each step for visualization

- ASCII/editor
  - Backend maps ASCII ↔ names via `char_encoder.py`; frontend does the same via `gridworks/src/lib/encoding.json`
  - The editor shows `basic_plant` using the wall tile, or a custom sprite if added
