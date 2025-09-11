#ifdef METTA_WITH_RAYLIB
#include "hermes.hpp"

#include <raylib.h>
#include <chrono>
#include <format>
#include <string>
#include <unordered_map>
#include <vector>

#include "action_handler.hpp"
#include "grid.hpp"
#include "grid_object.hpp"
#include "mettagrid_c.hpp"
#include "objects/agent.hpp"
#include "objects/converter.hpp"
#include "objects/wall.hpp"

#define HERMES_DEBUG 0

#if HERMES_DEBUG
#include <iostream>
#define HERMES_DBG(fmt, ...) std::cout << std::format(fmt##sv, __VA_ARGS__) << std::endl
#else
#define HERMES_DBG(fmt, ...)
#endif

#define DRAW_COUNTER(ctx) ++*ctx.num_draws

using std::literals::string_view_literals::operator""sv;

static inline bool is_zero(float a) {
  return std::abs(a) < 0.001f;
}

// Constants ------------------------------------------------------------------

static constexpr float MOVE_SPEED = 10.0f;
static constexpr float ZOOM_SPEED = 2.0f;
static constexpr float DRAG_SPEED = 1.0f;
static constexpr float SCROLL_SPEED = 0.1f;

static constexpr float TILE_SIZE = 200.0f;
static constexpr float INVENTORY_PADDING = 16.0f;

static constexpr Color VOID_COLOR = BLACK;
static constexpr Color FLOOR_COLOR = {0xCF, 0xA9, 0x70, 0xFF};
static constexpr Color PANEL_COLOR = {0, 0, 0, 0x80};

static constexpr uint32_t NO_SELECTION = std::numeric_limits<uint32_t>::max();

typedef uint8_t WallTile;
enum : WallTile {
  WallTile_0 = 0,
  WallTile_E = 1,
  WallTile_S = 2,
  WallTile_W = 4,
  WallTile_N = 8,

  WallTile_SE = WallTile_S | WallTile_E,
  WallTile_NW = WallTile_N | WallTile_W,

  WallTile_Fill = 16  // Special case for wall fill.
};

// Structures -----------------------------------------------------------------

// Serializable user configuration.
struct HermesConfig {
  static constexpr uint32_t V0 = 'M' | ('G' << 8) | ('H' << 16) | ('0' << 24);

  uint32_t signature;  // Indicates the version/features of the file format.
  int32_t monitor;
  Vector2 position;
  Camera2D camera;         // Also contains the window size in its offset parameter.
  GridObjectId selection;  // NO_SELECTION for none, otherwise object index.

  bool show_grid : 1;
  bool show_resources : 1;
  bool show_attack_mode : 1;
  bool show_fog_of_war : 1;
  bool show_visual_ranges : 1;
  bool show_ui : 1;
  bool show_help : 1;
  bool show_profiler : 1;
};

struct HermesActions {
  using Action = uint16_t;
  Action noop;
  Action move;
  Action rotate;

  bool show(Action action) const {
    return action != noop && action != move && action != rotate;
  }
};

// Resolved type names to batch indices for specific render nodes.
struct HermesTypes {
  uint8_t agent;
  uint8_t wall;
};

// Resolved sprite names to rectangle indices in the sprite sheet.
struct HermesSprites {
  using Sprite = uint16_t;
  std::vector<Sprite> objects;  // Indexed by type_name_id.
  std::vector<Sprite> actions;  // Indexed by action_name_id.
  std::vector<Sprite> items;    // Indexed by inventory_item_name_id.
  Sprite agent[4];              // Orientation sprite indices.
  Sprite wall[17];              // WallTile sprite indices.
  Sprite grid;
  Sprite halo;
  Sprite frozen;
  Sprite reward;
  Sprite selection;
  Sprite converting;
};

struct GridCell {
  GridCoord x, y;
};

// Lightweight data common to all render objects.
struct HermesNode {
  GridCell cell;
  uint16_t index : 16;  // Object id (for inventory), agent id, wall tile index
  uint8_t object : 4;   // Agent orientation/state, or color index.
  uint8_t output : 1;   // Converter has output resources
  uint8_t active : 1;   // Converter is converting
  uint8_t single : 1;   // Converter has a single output
  uint8_t unused : 1;
  uint8_t item_id;  // Item ID of the converter's single output
};
static_assert(sizeof(HermesNode) == 8);

// Simple profile metric, avoids making assumptions about performance.
struct HermesProfile {
  Color color;
  std::string_view name;
  std::chrono::microseconds time;
  std::chrono::high_resolution_clock::time_point start;
  uint32_t num_draws;
};

// Internal state, grouped by lifetimes.
struct Hermes {
  // Local resources (always present)
  std::string config_file;
  HermesConfig config;

  Vector2 last_mouse_pos;
  bool mouse_moved : 1;
  uint8_t padding : 6;
  bool initialized : 1;

  // Asset resources (lazily loaded on first render)
  std::unordered_map<std::string, HermesSprites::Sprite> sprite_lookup;
  std::vector<Rectangle> sprites;
  HermesSprites sprite_atlas;
  Texture2D sprite_sheet;

  // Scene resources (updated on episode start)
  const MettaGrid* next;
  const MettaGrid* self;
  const Grid* grid;

  HermesActions actions;
  HermesTypes types;
  std::vector<std::vector<HermesNode>> buckets;  // Also contains frame nodes.
  std::vector<GridCell> wall_fills;

  // Frame resources (updated on render)
  std::vector<uint16_t> frozen_agents;
  std::vector<float> rewards;
  std::vector<bool> visibility;
  GridObject* selection;  // Null if none or selected object invalid this frame.
  uint64_t items_mask;    // Bitmask of buckets whose nodes carry an inventory.
  uint64_t color_mask;    // Bitmask of buckets whose nodes carry a color index.
  uint64_t debug_mask;    // Bitmask of buckets the developer has chosen to display.

  // Statistics (for the UI) (num_agents stored in the MettaGrid environment)
  uint32_t num_walls;
  uint32_t num_objects;

  // Profiling (always running, optional display)
  HermesProfile setup;
  HermesProfile cache;
  std::vector<HermesProfile> profiles;
  uint32_t* num_draws;  // Counter for the render pass being profiled currently.

  bool show(uint32_t type_id) const {
    return (debug_mask & (1ull << type_id)) != 0;
  }
};

// Profiling helper to name, color and measure a block of code.
class HermesProfileScope {
  HermesProfile& _profile;

public:
  HermesProfileScope(Hermes& ctx, std::string_view name, Color color)
      : HermesProfileScope(ctx.profiles.emplace_back(), name, color) {
    ctx.num_draws = &_profile.num_draws;
  }

  HermesProfileScope(HermesProfile& profile, std::string_view name, Color color) : _profile(profile) {
    _profile.name = name;
    _profile.color = color;
    _profile.start = std::chrono::high_resolution_clock::now();
  }

  ~HermesProfileScope() {
    auto end = std::chrono::high_resolution_clock::now();
    _profile.time = std::chrono::duration_cast<std::chrono::microseconds>(end - _profile.start);
  }
};

// Helper to insert a profiling scope around an arbitrary statement.
#define HERMES_PROFILE(ctx, name, color, run)    \
  do {                                           \
    HermesProfileScope _scope(ctx, name, color); \
    run;                                         \
  } while (false)

// Render Passes --------------------------------------------------------------

// HermesNode -> Raylib world position.
static inline Vector2 Position(HermesNode node) {
  return {node.cell.x * TILE_SIZE, node.cell.y * TILE_SIZE};
}

// HermesNode -> Raylib world rectangle, sized to the grid itself.
static inline Rectangle TileRect(GridCell cell) {
  return {cell.x * TILE_SIZE, cell.y * TILE_SIZE, TILE_SIZE, TILE_SIZE};
}
static inline Rectangle TileRect(HermesNode node) {
  return TileRect(node.cell);
}

// Heavy lifter, and the most expensive function to call in the whole renderer.
static inline void Draw(Hermes& ctx, Rectangle sprite, Rectangle rect, float rot = 0, Color color = WHITE) {
  Vector2 pivot = {rect.width / 2, rect.height / 2};
  DrawTexturePro(ctx.sprite_sheet, sprite, rect, pivot, rot, color);
  DRAW_COUNTER(ctx);
}
static inline void Draw(Hermes& ctx, Rectangle sprite, Vector2 pos, float rot = 0, Color color = WHITE) {
  Draw(ctx, sprite, {pos.x, pos.y, sprite.width, sprite.height}, rot, color);
}

static void DrawFloor(Hermes& ctx) {
  constexpr auto sz = static_cast<int>(TILE_SIZE);
  DrawRectangle(-sz / 2, -sz / 2, ctx.grid->width * sz, ctx.grid->height * sz, FLOOR_COLOR);
  DRAW_COUNTER(ctx);
}

static void DrawWalls(Hermes& ctx) {
  if (!ctx.show(ctx.types.wall)) {
    return;
  }

  for (auto node : ctx.buckets[ctx.types.wall]) {
    Draw(ctx, ctx.sprites[ctx.sprite_atlas.wall[node.index]], TileRect(node));
  }

  auto fill_sprite = ctx.sprites[ctx.sprite_atlas.wall[WallTile_Fill]];
  for (auto cell : ctx.wall_fills) {
    auto pos = TileRect(cell);
    pos.x += TILE_SIZE / 2;
    pos.y += TILE_SIZE / 2 - 42;
    Draw(ctx, fill_sprite, pos);
  }
}

static void DrawTrajectory(Hermes& ctx) {
  if (ctx.selection == nullptr) {
    return;
  }

  // TODO capture frame data across time to support drawing this.
}

static void DrawObjects(Hermes& ctx) {
  auto t = 0u;
  for (const auto& bucket : ctx.buckets) {
    auto type_id = t++;
    if (type_id != ctx.types.wall && type_id != ctx.types.agent && ctx.show(type_id)) {
      auto sprite = ctx.sprites[ctx.sprite_atlas.objects[type_id]];
      for (auto node : bucket) {
        Draw(ctx, sprite, Position(node));
      }
    }
  }
}

static void DrawAgents(Hermes& ctx) {
  if (!ctx.show(ctx.types.agent)) {
    return;
  }

  Color color;
  HermesSprites::Sprite sprite;
  for (auto node : ctx.buckets[ctx.types.agent]) {
    sprite = ctx.sprite_atlas.agent[node.object];
    color = {static_cast<uint8_t>(fmodf(node.index * static_cast<float>(M_PI), 1.0f) * 0xFF),
             static_cast<uint8_t>(fmodf(node.index * static_cast<float>(M_E), 1.0f) * 0xFF),
             static_cast<uint8_t>(fmodf(node.index * static_cast<float>(M_SQRT2), 1.0f) * 0xFF),
             0xFF};
    Draw(ctx, ctx.sprites[sprite], Position(node), 0, color);
  }
}

static inline float ToDegrees(Orientation orientation, Vector2& offset) {
  switch (orientation) {
    default:
    case Orientation::East:
      offset.x = 1;
      offset.y = 0;
      return 0;
    case Orientation::West:
      offset.x = -1;
      offset.y = 0;
      return 180;
    case Orientation::North:
      offset.x = 0;
      offset.y = -1;
      return 270;
    case Orientation::South:
      offset.x = 0;
      offset.y = 1;
      return 90;
  }
}

static void DrawActions(Hermes& ctx) {
  if (!ctx.show(ctx.types.agent)) {
    return;
  }

  auto actions = ctx.self->actions();
  if (actions.ndim() == 2) {
    auto actions_view = actions.unchecked<2>();
    auto action_success = ctx.self->action_success();
    const auto& agents = ctx.buckets[ctx.types.agent];
    for (auto i = 0u; i < agents.size(); i++) {
      auto action = static_cast<uint16_t>(actions_view(i, 0));
      if (action_success[i] && ctx.actions.show(action)) {
        Vector2 ofs;
        auto node = agents[i];
        auto rot = ToDegrees(static_cast<Orientation>(node.object), ofs);
        auto pos = Position(node);
        pos.x += ofs.x * TILE_SIZE / 2;
        pos.y += ofs.y * TILE_SIZE / 2;
        Draw(ctx, ctx.sprites[ctx.sprite_atlas.actions[action]], pos, rot);
      }
    }
  }

  auto sprite = ctx.sprites[ctx.sprite_atlas.frozen];
  const auto& agents = ctx.buckets[ctx.types.agent];
  for (auto agent_id : ctx.frozen_agents) {
    auto node = agents[agent_id];
    Draw(ctx, sprite, Position(node));
  }
}

static void DrawSelection(Hermes& ctx) {
  if (ctx.selection == nullptr) {
    return;
  }

  auto loc = ctx.selection->location;
  HermesNode node = {.cell = {.x = loc.r, .y = loc.c}};
  Draw(ctx, ctx.sprites[ctx.sprite_atlas.selection], Position(node));
}

static void DrawInventory(Hermes& ctx) {
  if (!ctx.config.show_resources) {
    return;
  }

  Rectangle dst;
  auto set_position = [&dst](HermesNode node, float offset = 16) {
    auto pos = Position(node);
    dst.x = pos.x - TILE_SIZE / 2;
    dst.y = pos.y - TILE_SIZE / 2 + offset;
  };

  auto draw = [&ctx, &dst](size_t item_id, float scale = 8) {
    auto sprite = ctx.sprites[ctx.sprite_atlas.items[item_id]];
    dst.width = sprite.width / scale;
    dst.height = sprite.height / scale;
    Draw(ctx, sprite, dst);
  };

  using Inventory = std::map<InventoryItem, InventoryQuantity>;
  auto multi_draw = [draw, &dst](const Inventory& inv) {
    auto advanceX = std::min(32.0f, (TILE_SIZE - INVENTORY_PADDING * 2) / inv.size());
    for (auto [item_id, quantity] : inv) {
      if (quantity > 0) {
        draw(item_id);
        dst.x += advanceX;
      }
    }
  };

  auto t = 0u;
  for (const auto& bucket : ctx.buckets) {
    auto type_id = t++;
    if ((ctx.items_mask & (1 << type_id)) == 0 || !ctx.show(type_id)) {
      continue;
    }

    for (auto node : bucket) {
      const auto& inv = static_cast<HasInventory*>(ctx.grid->object(node.index))->inventory;
      if (inv.size() == 0) {
        continue;
      }

      if (inv.size() == 1 && node.single) {
        auto kv = inv.begin();
        auto id = kv->first;
        if (id == node.item_id && kv->second == 1) {
          set_position(node, 8);
          draw(id, 2);
          continue;
        }
      }

      set_position(node);
      multi_draw(inv);
    }
  }

  if (ctx.show(ctx.types.agent)) {
    for (const auto node : ctx.buckets[ctx.types.agent]) {
      const auto& inv = ctx.self->agent(node.index)->inventory;
      if (inv.size() != 0) {
        set_position(node);
        multi_draw(inv);
      }
    }
  }
}

static void DrawRewards(Hermes& ctx) {
  if (!ctx.show(ctx.types.agent)) {
    return;
  }

  auto node_id = 0u;
  auto sprite = ctx.sprites[ctx.sprite_atlas.reward];
  auto width = sprite.width / 8;
  auto height = sprite.height / 8;
  for (auto node : ctx.buckets[ctx.types.agent]) {
    auto reward = ctx.rewards[node_id++];
    auto advanceX = std::min(32.0f, TILE_SIZE / reward);
    auto pos = Position(node);
    Rectangle dst = {pos.x - TILE_SIZE / 2, pos.y + TILE_SIZE / 2 - 16, width, height};
    for (auto i = 0; i < reward; i++) {
      Draw(ctx, sprite, dst);
      dst.x += advanceX;
    }
  }
}

static void DrawVisibility(Hermes& ctx) {
  if (!ctx.config.show_visual_ranges && !ctx.config.show_fog_of_war) {
    return;
  }

  std::fill(ctx.visibility.begin(), ctx.visibility.end(), false);

  uint32_t grid_width = ctx.grid->width;
  uint32_t grid_height = ctx.grid->height;
  uint32_t obs_width = ctx.self->obs_width;
  uint32_t obs_height = ctx.self->obs_height;
  auto view_width = obs_width / 2;
  auto view_height = obs_height / 2;

  auto update_visibility = [=, &ctx](uint32_t agent_x, uint32_t agent_y) {
    auto x1 = std::max(agent_x - view_width, 0u);
    auto y1 = std::max(agent_y - view_height, 0u);
    auto x2 = std::min(x1 + obs_width, grid_width);
    auto y2 = std::min(y1 + obs_height, grid_height);
    for (; y1 < y2; y1++) {
      auto y = y1 * grid_width;
      for (auto x = x1; x < x2; x++) {
        ctx.visibility[y + x] = true;
      }
    }
  };

  if (ctx.selection != nullptr && ctx.selection->type_id == ctx.types.agent) {
    auto loc = ctx.selection->location;
    update_visibility(loc.r, loc.c);
  } else {
    for (auto agent : ctx.buckets[ctx.types.agent]) {
      update_visibility(agent.cell.x, agent.cell.y);
    }
  }

  constexpr auto pivot = TILE_SIZE / 2;  // Match sprites having a centered pivot.
  Rectangle rect = {0, 0, TILE_SIZE, TILE_SIZE};
  Color color = {0, 0, 0, static_cast<uint8_t>(ctx.config.show_fog_of_war ? 0xFF : 0x20)};
  for (auto y = 0u; y < grid_height; y++) {
    rect.y = y * TILE_SIZE - pivot;
    for (auto x = 0u; x < grid_width; x++) {
      if (!ctx.visibility[y * grid_width + x]) {
        rect.x = x * TILE_SIZE - pivot;
        DrawRectangleRec(rect, color);
        DRAW_COUNTER(ctx);
      }
    }
  }
}

static void DrawGrid(Hermes& ctx) {
  if (!ctx.config.show_grid) {
    return;
  }

  Rectangle pos = {0, 0, TILE_SIZE, TILE_SIZE};
  auto grid_width = ctx.grid->width;
  auto grid_height = ctx.grid->height;
  auto grid_sprite = ctx.sprites[ctx.sprite_atlas.grid];
  for (auto y = 0; y < grid_height; y++) {
    pos.y = y * TILE_SIZE;
    for (auto x = 0; x < grid_width; x++) {
      pos.x = x * TILE_SIZE;
      Draw(ctx, grid_sprite, pos);
    }
  }
}

static void DrawThoughtBubbles(Hermes& ctx) {
  if (ctx.selection == nullptr || ctx.selection->type_id != ctx.types.agent) {
    return;
  }

  // TODO need to capture replay data to support this
  // auto agent = static_cast<Agent*>(ctx.selection);
}

static void DrawAttackMode(Hermes& ctx) {
  if (!ctx.config.show_attack_mode) {
    return;
  }

  // TODO
}

static void DrawPanel(Hermes& ctx, int x, int y, int w, int h) {
  constexpr auto pad = 5;
  DrawRectangle(x - pad, y - pad, w + pad * 2, h + pad * 2, PANEL_COLOR);
  DRAW_COUNTER(ctx);
}

static void DrawUI(Hermes& ctx) {
  constexpr auto size = 20;
  if (ctx.config.show_ui) {
    char buffer[256];
    auto num_agents = static_cast<int>(ctx.buckets[ctx.types.agent].size());
    snprintf(buffer,
             sizeof(buffer),
             "Agents: %d, Walls: %d, Objects: %d, Step: %d",
             num_agents,
             ctx.num_walls,
             ctx.num_objects,
             ctx.self->current_step);

    DrawPanel(ctx, 10, 10, 500, size);
    DrawText(buffer, 10, 10, size, RAYWHITE);
    DRAW_COUNTER(ctx);
  }

  if (ctx.config.show_help) {
    auto x = GetScreenWidth() - 255;
    auto y = GetScreenHeight() - 7 * size;
    DrawPanel(ctx, x, y, 250, 7 * size);
    auto draw = [&](const char* line) {
      DrawText(line, x, y, size, RAYWHITE);
      DRAW_COUNTER(ctx);
      y += size;
    };
    draw("F1: Show Help");
    draw("F2: Show UI");
    draw("F3: Show Profiler");
    draw("G: Show Grid");
    draw("R: Show Resources");
    // draw("A: Show Attack Mode");
    draw("F: Show Fog of War");
    draw("V: Show Visual Ranges");
  }

  if (ctx.config.show_profiler) {
    for (auto i = 0; i < 64; i++) {
      if (ctx.show(static_cast<uint32_t>(i))) {
        DrawRectangle(10 + i * 6, 36, 4, 10, RAYWHITE);
        DRAW_COUNTER(ctx);
      }
    }
  }
}

static void DrawProfiler(Hermes& ctx) {
  if (ctx.config.show_profiler) {
    constexpr auto size = 14;
    constexpr auto bar = 3;
    constexpr auto pad = 5;
    constexpr auto w = 300;
    auto x = 10;
    auto y = 70;
    DrawPanel(ctx, x, y, w, (3 + static_cast<int>(ctx.profiles.size())) * (size + bar + pad) + size);

    char buffer[256];
    auto draw = [&](const HermesProfile& profile) {
      auto calls = profile.num_draws;
      auto time_us = static_cast<int32_t>(profile.time.count());
      auto average = calls == 0 ? 0.0 : static_cast<double>(time_us) / calls;
      snprintf(buffer, sizeof(buffer), "%4d draws (%.2f) - %3d us", calls, average, time_us);
      DrawText(profile.name.data(), x, y, size, profile.color);
      DrawText(buffer, x + 100, y, size, profile.color);
      DrawRectangle(x, y + size, std::min(w, time_us / 4), bar, profile.color);
      y += size + bar + pad;
    };
    draw(ctx.setup);
    draw(ctx.cache);
    int64_t frame_time = 0;
    for (const auto& profile : ctx.profiles) {
      draw(profile);
      frame_time += profile.time.count();
    }
    auto frame = frame_time / 1000.0;
    auto color = frame > 16.666 ? RED : GREEN;
    snprintf(buffer, sizeof(buffer), "Frame: %.2f ms", frame);
    DrawText(buffer, x, y + size, size, color);
  }
  ctx.profiles.clear();
}

// Frame Rendering ------------------------------------------------------------

static void Hermes_Close(Hermes& ctx) {
  ctx.config.monitor = GetCurrentMonitor();
  ctx.config.position = GetWindowPosition();

  auto file = ctx.config_file.c_str();
  if (file && !SaveFileData(file, &ctx.config, sizeof(ctx.config))) {
    fprintf(stderr, "Failed to save hermes config file %.*s\n", static_cast<int>(ctx.config_file.size()), file);
  }

  UnloadTexture(ctx.sprite_sheet);
  CloseWindow();

  ctx.initialized = false;
}

#define Draw(f, color) HERMES_PROFILE(ctx, #f, (color), Draw##f(ctx))

static void Hermes_Image(Hermes& ctx) {
  if (WindowShouldClose()) {
    Hermes_Close(ctx);
    return;
  }

  BeginDrawing();
  ClearBackground(VOID_COLOR);
  BeginMode2D(ctx.config.camera);
  Draw(Floor, WHITE);
  Draw(Walls, WHITE);
  Draw(Trajectory, WHITE);
  Draw(Objects, WHITE);
  Draw(Agents, WHITE);
  Draw(Actions, WHITE);
  Draw(Selection, WHITE);
  Draw(Inventory, WHITE);
  Draw(Rewards, WHITE);
  Draw(Visibility, WHITE);
  Draw(Grid, WHITE);
  Draw(ThoughtBubbles, WHITE);
  Draw(AttackMode, WHITE);
  EndMode2D();
  Draw(UI, BLUE);
  DrawProfiler(ctx);
  EndDrawing();
}

#undef Draw

// Breaks down the scene into independent buckets for each dynamic object type.
// This traverses the MettaGrid instance once per frame and simplifies drawing.
//
// Note that parts of the MettaGrid touched by single draw calls are not batched.
static void Hermes_Batch(Hermes& ctx) {
  HermesProfileScope _hps(ctx, "Batch"sv, ORANGE);

  // Clear all buckets, except for those containing static scene objects.
  auto type_id = 0;
  for (auto& bucket : ctx.buckets) {
    if (type_id++ == ctx.types.wall) {
      continue;
    }

    bucket.clear();
  }

  auto num_agents = ctx.self->num_agents();
  ctx.frozen_agents.clear();
  ctx.frozen_agents.reserve(num_agents);
  ctx.rewards.clear();
  ctx.rewards.reserve(num_agents);

  // Extract scene data from changing objects into buckets matching their type_id.
  auto count = 0u;
  for (const auto& ptr : ctx.grid->objects) {
    const auto obj = ptr.get();
    if (obj == nullptr) {
      continue;
    }

    auto loc = obj->location;
    HermesNode node = {.cell = {.x = loc.r, .y = loc.c}};

    auto t = obj->type_id;
    if (t == ctx.types.wall) {
      continue;  // Walls are batched once per episode.
    } else if (t == ctx.types.agent) {
      const auto agent = static_cast<Agent*>(obj);
      node.index = static_cast<uint16_t>(agent->agent_id);
      node.object = agent->orientation;
      ctx.rewards.push_back(*agent->reward);
      if (agent->frozen > 0) {
        ctx.frozen_agents.push_back(static_cast<uint16_t>(ctx.buckets[t].size()));
      }
    } else if (ctx.items_mask & (1ull << t)) {
      node.index = static_cast<uint16_t>(obj->id);
      if (ctx.color_mask & (1ull << t)) {
        const auto con = static_cast<Converter*>(obj);
        const auto& res = con->output_resources;
        auto color = con->color;
        node.object = color > 2 ? 0 : color;
        node.active = con->converting;
        node.output = res.size() > 0;
        node.single = res.size() == 1;
        node.item_id = node.single ? res.begin()->first : 0;
      }
    }

    ctx.buckets[t].push_back(node);
    count++;
  }

  ctx.num_objects = count;

  // Update the selection pointer each frame, because the object may have been removed.
  ctx.selection = ctx.config.selection != NO_SELECTION ? ctx.grid->objects[ctx.config.selection].get() : nullptr;
}

static void Hermes_SetupCamera(Hermes& ctx) {
  ctx.config.camera.offset = {GetScreenWidth() / 2.0f, GetScreenHeight() / 2.0f};
}

static void Hermes_ClampCameraZoom(Hermes& ctx) {
  auto zoom_x = GetScreenWidth() / (ctx.grid->width * TILE_SIZE) / 2;
  auto zoom_y = GetScreenHeight() / (ctx.grid->height * TILE_SIZE) / 2;
  auto& zoom = ctx.config.camera.zoom;
  zoom = std::clamp(zoom, std::min(zoom_x, zoom_y), 2.5f);
}

static void Hermes_ClampCameraTarget(Hermes& ctx) {
  constexpr auto pad = TILE_SIZE / 2;
  auto zoom = ctx.config.camera.zoom;
  auto& target = ctx.config.camera.target;
  target.x = std::clamp(target.x, -pad, ctx.grid->width * TILE_SIZE - GetScreenWidth() * zoom / 2 - pad);
  target.y = std::clamp(target.y, -pad, ctx.grid->height * TILE_SIZE - GetScreenHeight() * zoom / 2 - pad);
}

// Handles user events and updates the Hermes configuration or camera in response.
static void Hermes_Input(Hermes& ctx) {
  HermesProfileScope _hps(ctx, "Input"sv, GREEN);

  // Window + camera = viewport.
  if (IsWindowResized()) {
    Hermes_SetupCamera(ctx);
  }

// Configuration toggles.
#define CONFIG(k, v) \
  if (IsKeyPressed(k)) ctx.config.v = !ctx.config.v
  CONFIG(KEY_G, show_grid);
  CONFIG(KEY_R, show_resources);
  CONFIG(KEY_M, show_attack_mode);
  CONFIG(KEY_F, show_fog_of_war);
  CONFIG(KEY_V, show_visual_ranges);
  CONFIG(KEY_F1, show_help);
  CONFIG(KEY_F2, show_ui);
  CONFIG(KEY_F3, show_profiler);
#undef CONFIG

  // Clearing selection.
  if (IsKeyPressed(KEY_ESCAPE)) {
    ctx.config.selection = NO_SELECTION;
  }

  // Cycling selection through agents.
  if (IsKeyPressed(KEY_TAB)) {
    const auto& objects = ctx.grid->objects;
    auto num = objects.size();
    auto idx = ctx.config.selection == NO_SELECTION ? 0 : ctx.config.selection + 1;
    for (; idx < num; idx++) {
      auto obj = objects[idx].get();
      if (obj != nullptr && obj->type_id == ctx.types.agent) {
        ctx.config.selection = idx;
        break;
      }
    }
    if (idx == num) {
      ctx.config.selection = NO_SELECTION;
    }
  }

  // Mouse camera controls.
  auto zoom = GetMouseWheelMove() * SCROLL_SPEED;
  auto mouse_pos = GetMousePosition();
  auto delta_time = GetFrameTime();
  auto delta_scale = ctx.config.camera.zoom;
  auto& target = ctx.config.camera.target;
  if (IsMouseButtonDown(MOUSE_BUTTON_LEFT)) {
    auto delta_x = mouse_pos.x - ctx.last_mouse_pos.x;
    auto delta_y = mouse_pos.y - ctx.last_mouse_pos.y;
    if (!is_zero(delta_x) || !is_zero(delta_y)) {
      target.x -= delta_x / delta_scale * DRAG_SPEED;
      target.y -= delta_y / delta_scale * DRAG_SPEED;
      ctx.mouse_moved = true;
    }
  }
  if (IsMouseButtonReleased(MOUSE_BUTTON_LEFT)) {
    if (!ctx.mouse_moved) {
      auto world_pos = GetScreenToWorld2D(mouse_pos, ctx.config.camera);
      auto grid_x = static_cast<GridCoord>((world_pos.x + TILE_SIZE / 2) / TILE_SIZE);
      auto grid_y = static_cast<GridCoord>((world_pos.y + TILE_SIZE / 2) / TILE_SIZE);
      if (grid_x >= 0 && grid_x < ctx.grid->width && grid_y >= 0 && grid_y < ctx.grid->height) {
        Layer layer = GridLayer::GridLayerCount;
        do {
          auto obj = ctx.grid->object_at({grid_x, grid_y, --layer});
          if (obj != nullptr) {
            HERMES_DBG("Pick: {}x{}.{} {}", grid_x, grid_y, layer, obj->id);
            ctx.config.selection = obj->id;
            break;
          }
        } while (layer != 0);
      }
    }
    ctx.mouse_moved = false;
  }
  ctx.last_mouse_pos = mouse_pos;

// Keyboard camera controls.
#define MOVE(k, d, dir) \
  if (IsKeyDown(k)) target.d += delta_time / delta_scale * dir * MOVE_SPEED * TILE_SIZE
  MOVE(KEY_W, y, -1);
  MOVE(KEY_S, y, +1);
  MOVE(KEY_A, x, -1);
  MOVE(KEY_D, x, +1);
#undef MOVE

#define ZOOM(k, dir) \
  if (IsKeyDown(k)) zoom = delta_time * dir * ZOOM_SPEED
  ZOOM(KEY_Q, +1);
  ZOOM(KEY_E, -1);
#undef ZOOM

  if (is_zero(zoom)) {
    Hermes_ClampCameraTarget(ctx);
  } else {
    auto before = GetScreenToWorld2D(mouse_pos, ctx.config.camera);
    ctx.config.camera.zoom *= 1.0f + zoom;
    Hermes_ClampCameraZoom(ctx);

    auto after = GetScreenToWorld2D(mouse_pos, ctx.config.camera);
    target.x += before.x - after.x;
    target.y += before.y - after.y;
    Hermes_ClampCameraTarget(ctx);
  }

  auto base = 0u;
  if (IsKeyDown(KEY_LEFT_SHIFT)) base += 10;
  if (IsKeyDown(KEY_LEFT_CONTROL)) base += 20;
  if (IsKeyDown(KEY_LEFT_ALT)) base += 40;
  for (auto i = 0; i < 10; i++) {
    auto offset = base + static_cast<uint32_t>(i);
    if (offset < 64u && IsKeyPressed(KEY_ZERO + i)) {
      ctx.debug_mask ^= 1ull << offset;
      HERMES_DBG("Debug: {} = {}", offset, ctx.show(offset));
    }
  }
}

// Prepares the scene into a set of static resources reused for the entire episode.
static void Hermes_Cache(Hermes& ctx) {
  if (ctx.self == ctx.next) {
    return;
  }

  HermesProfileScope _hps(ctx.cache, "Cache"sv, RED);

  ctx.self = ctx.next;
  ctx.grid = &ctx.self->grid();

  if (ctx.config.selection >= ctx.grid->objects.size()) {
    ctx.config.selection = NO_SELECTION;
  }

  size_t grid_width = ctx.grid->width;
  size_t grid_height = ctx.grid->height;
  HERMES_DBG("Grid: {}x{}", grid_width, grid_height);

  // Map object type names to sprite indices.
  const auto& type_names = ctx.self->object_type_names;
  ctx.sprite_atlas.objects.resize(type_names.size());

  uint8_t type_id = 0;
  for (const auto& name : type_names) {
    HermesSprites::Sprite sprite = 0;
    if (!name.empty()) {
      sprite = ctx.sprite_lookup[std::format("objects/{}.png"sv, name)];
      HERMES_DBG("Object: {} = Sprite {}", name, sprite);

      if (name == "agent"sv) {
        ctx.types.agent = type_id;
      } else if (name == "wall"sv) {
        ctx.types.wall = type_id;
      }
    }

    ctx.sprite_atlas.objects[type_id++] = sprite;
  }

  HERMES_DBG("Agent.type_id: {}", static_cast<int>(ctx.types.agent));
  HERMES_DBG("Wall.type_id: {}", static_cast<int>(ctx.types.wall));

  // Map action names to sprite indices.
  const auto& action_handlers = ctx.self->action_handlers();
  ctx.sprite_atlas.actions.resize(action_handlers.size());

  auto action_id = 0u;
  for (const auto& handler : action_handlers) {
    const auto& name = handler->action_name();

#define ACTION(s) else if (name == #s##sv) ctx.actions.s = static_cast<uint8_t>(action_id)
    if (false) {
      // Intentional no-op: anchor for ACTION macro chain
    }
    ACTION(noop);
    ACTION(move);
    ACTION(rotate);
#undef ACTION

    auto file = std::format("actions/icons/{}.png", name);
    ctx.sprite_atlas.actions[action_id++] = ctx.sprite_lookup[file];
    HERMES_DBG("Action: {} = Sprite {}", name, ctx.sprite_lookup[file]);
  }

  // Map inventory item names to sprite indices.
  const auto& item_names = ctx.self->resource_names;
  ctx.sprite_atlas.items.resize(item_names.size());

  auto item_id = 0u;
  for (const auto& name : item_names) {
    auto file = std::format("resources/{}.png", name);
    ctx.sprite_atlas.items[item_id++] = ctx.sprite_lookup[file];
    HERMES_DBG("Inventory: {} = Sprite {}", name, ctx.sprite_lookup[file]);
  }

  // Setup buckets for each object type.
  ctx.buckets.resize(type_names.size());
  ctx.buckets[ctx.types.wall].clear();
  ctx.wall_fills.clear();

  // Extract scene data from unchanging objects.
  uint64_t items_mask = 0;
  uint64_t color_mask = 0;
  auto& walls = ctx.buckets[ctx.types.wall];
  for (const auto& ptr : ctx.grid->objects) {
    const auto obj = ptr.get();
    if (obj == nullptr) {
      continue;
    }
    if (obj->type_id >= type_names.size()) {
      HERMES_DBG("Object: {} = Type ID {}", obj->type_name, obj->type_id);
    } else if (obj->type_id == ctx.types.wall) {
      auto loc = obj->location;
      walls.push_back({.cell = {.x = loc.r, .y = loc.c}});
    } else if (auto inv = dynamic_cast<HasInventory*>(obj)) {
      items_mask |= 1 << obj->type_id;
      if (dynamic_cast<Converter*>(inv)) {
        color_mask |= 1 << obj->type_id;
      }
    }
  }
  ctx.items_mask = items_mask;
  ctx.color_mask = color_mask;
  ctx.num_walls = static_cast<uint32_t>(walls.size());

  // Construct a wall adjacency map and assign each wall node a sprite index.
  auto wall_map_width = grid_width + 2;      // Surround the grid with empty space to
  auto wall_map_height = grid_height + 2;    // avoid testing for borders at the edges.
  std::vector<bool> wall_del(walls.size());  // Mark wall nodes covered by wall fills.
  std::vector<bool> wall_map(wall_map_width * wall_map_height);
  for (auto wall : walls) {
    wall_map[(wall.cell.y + 1) * wall_map_width + wall.cell.x + 1] = true;
  }
  auto wall_idx = 0u;
  for (auto& wall : walls) {
    auto x = wall.cell.x + 1u;
    auto y = wall.cell.y + 1u;

#define WALL(x, y) wall_map[(y) * wall_map_width + (x)]
    WallTile tile = WallTile_0;
    if (WALL(x, y + 1)) tile |= WallTile_S;
    if (WALL(x + 1, y)) tile |= WallTile_E;
    if (WALL(x, y - 1)) tile |= WallTile_N;
    if (WALL(x - 1, y)) tile |= WallTile_W;
    wall.index = tile;

    if ((tile & WallTile_SE) == WallTile_SE && WALL(x + 1, y + 1)) {
      ctx.wall_fills.push_back({.x = wall.cell.x, .y = wall.cell.y});

      if ((tile & WallTile_NW) == WallTile_NW && WALL(x + 1, y - 1) && WALL(x - 1, y - 1) && WALL(x - 1, y + 1)) {
        wall_del[wall_idx] = true;
      }
    }

    wall_idx++;
#undef WALL
  }

  // Don't render walls covered by wall fills.
  auto write_idx = 0u;
  auto read_idx = 0u;
  for (; read_idx < walls.size(); read_idx++) {
    if (!wall_del[read_idx]) {
      if (write_idx != read_idx) {
        walls[write_idx] = walls[read_idx];
      }
      write_idx++;
    }
  }
  assert(wall_del.size() == read_idx - write_idx);
  walls.resize(write_idx);

  // Intermediate frame memory whose size doesn't change between frames.
  ctx.visibility.resize(grid_width * grid_height);

  // Validate camera settings against the new scene.
  Hermes_ClampCameraZoom(ctx);  // Requires grid size. Target is validated every frame.
}

// Initializes the Hermes instance, its Raylib window, and sprite sheet.
static void Hermes_Setup(Hermes& ctx) {
  if (ctx.initialized) {
    return;
  }

  HermesProfileScope _hps(ctx.setup, "Setup", MAROON);

  // Query the environment through our helper python module.
  auto hermes_py = py::module::import("metta.mettagrid.renderer.hermes");
  auto asset_path = hermes_py.attr("get_asset_path")().cast<std::string>();
  auto config_path = hermes_py.attr("get_config_path")().cast<std::string>();
  HERMES_DBG("Asset path: {}", asset_path);
  HERMES_DBG("Config file: {}", config_path);

  // Load the user file, if it exists, to preserve configuration across sessions.
  ctx.config_file = std::format("{}/hermes.bin", config_path);

  int bin_size;
  bool has_config = false;
  if (auto bin = LoadFileData(ctx.config_file.c_str(), &bin_size)) {
    auto ptr = static_cast<HermesConfig*>(static_cast<void*>(bin));
    if (bin_size == sizeof(HermesConfig) && ptr->signature == HermesConfig::V0) {
      ctx.config = *ptr;
      has_config = true;
    }
    UnloadFileData(bin);
  }

  // Raylib initialization.
  SetConfigFlags(FLAG_WINDOW_RESIZABLE);
  InitWindow(1280, 960, "Lens of Hermes");
  SetExitKey(KEY_NULL);
  SetTargetFPS(60);

  // Sprite sheet texture.
  ctx.sprite_sheet = LoadTexture(std::format("{}/atlas.png"sv, asset_path).c_str());
  GenTextureMipmaps(&ctx.sprite_sheet);
  SetTextureFilter(ctx.sprite_sheet, TEXTURE_FILTER_TRILINEAR);
  SetTextureWrap(ctx.sprite_sheet, TEXTURE_WRAP_CLAMP);

  // Sprite sheet atlas and lookup.
  auto builtins = py::module::import("builtins");
  auto json = py::module::import("json");
  auto config = json.attr("load")(builtins.attr("open")(std::format("{}/atlas.json", asset_path), "r"sv));

  auto num_sprites = py::len(config);
  ctx.sprite_lookup.reserve(num_sprites);
  ctx.sprites.reserve(num_sprites);

  for (auto& [key, value] : config.cast<py::dict>()) {
    auto name = key.cast<std::string>();
    auto sprite = value.cast<py::list>();
    ctx.sprite_lookup[name] = static_cast<HermesSprites::Sprite>(ctx.sprites.size());
    ctx.sprites.emplace_back(Rectangle{.x = sprite[0].cast<float>(),
                                       .y = sprite[1].cast<float>(),
                                       .width = sprite[2].cast<float>(),
                                       .height = sprite[3].cast<float>()});
  }

#if HERMES_DEBUG
  std::vector<std::string> keys;
  keys.reserve(ctx.sprite_lookup.size());
  for (const auto& kv : ctx.sprite_lookup) keys.push_back(kv.first);
  std::sort(keys.begin(), keys.end());
  for (const auto& k : keys)
    HERMES_DBG(
        "Sprite: {} ({}x{})", k, ctx.sprites[ctx.sprite_lookup[k]].width, ctx.sprites[ctx.sprite_lookup[k]].height);
#endif

  // Predefined sprites.
  auto agent_sprite = [&ctx](std::string_view suffix, Orientation orientation) {
    auto file = std::format("agents/agent.{}.png"sv, suffix);
    ctx.sprite_atlas.agent[orientation] = ctx.sprite_lookup[file];
    HERMES_DBG("Agent {} = {}", suffix, ctx.sprite_lookup[file]);
  };

  auto wall_sprite = [&ctx](std::string_view suffix, WallTile tile) {
    auto file = std::format("objects/wall.{}.png"sv, suffix);
    ctx.sprite_atlas.wall[tile] = ctx.sprite_lookup[file];
    HERMES_DBG("Wall {} = {}", suffix, ctx.sprite_lookup[file]);
  };

  agent_sprite("n", Orientation::North);
  agent_sprite("s", Orientation::South);
  agent_sprite("w", Orientation::West);
  agent_sprite("e", Orientation::East);
  ctx.sprite_atlas.frozen = ctx.sprite_lookup["agents/frozen.png"];

  wall_sprite("0", WallTile_0);
  wall_sprite("e", WallTile_E);
  wall_sprite("s", WallTile_S);
  wall_sprite("se", WallTile_S | WallTile_E);
  wall_sprite("w", WallTile_W);
  wall_sprite("we", WallTile_W | WallTile_E);
  wall_sprite("ws", WallTile_W | WallTile_S);
  wall_sprite("wse", WallTile_W | WallTile_S | WallTile_E);
  wall_sprite("n", WallTile_N);
  wall_sprite("ne", WallTile_N | WallTile_E);
  wall_sprite("ns", WallTile_N | WallTile_S);
  wall_sprite("nse", WallTile_N | WallTile_S | WallTile_E);
  wall_sprite("nw", WallTile_N | WallTile_W);
  wall_sprite("nwe", WallTile_N | WallTile_W | WallTile_E);
  wall_sprite("nws", WallTile_N | WallTile_W | WallTile_S);
  wall_sprite("nwse", WallTile_N | WallTile_W | WallTile_S | WallTile_E);
  wall_sprite("fill", WallTile_Fill);

  ctx.sprite_atlas.grid = ctx.sprite_lookup["view/grid.png"];
  ctx.sprite_atlas.halo = ctx.sprite_lookup["effects/halo.png"];
  ctx.sprite_atlas.reward = ctx.sprite_lookup["resources/reward.png"];
  ctx.sprite_atlas.selection = ctx.sprite_lookup["selection.png"];
  ctx.sprite_atlas.converting = ctx.sprite_lookup["actions/converting.png"];

  // Scene initialization.
  constexpr auto min_width = 400;
  constexpr auto min_height = 300;
  SetWindowMinSize(min_width, min_height);
  if (has_config) {
    if (ctx.config.monitor < 0 || ctx.config.monitor >= GetMonitorCount()) {
      ctx.config.monitor = 0;
    }
    SetWindowMonitor(ctx.config.monitor);

    auto mon_w = GetMonitorWidth(ctx.config.monitor);
    auto mon_h = GetMonitorHeight(ctx.config.monitor);
    auto x = std::clamp(static_cast<int>(ctx.config.position.x), 0, mon_w - min_width);
    auto y = std::clamp(static_cast<int>(ctx.config.position.y), 0, mon_h - min_height);
    SetWindowPosition(x, y);

    auto o = ctx.config.camera.offset;
    auto w = std::clamp(static_cast<int>(o.x * 2), min_width, mon_w - x);
    auto h = std::clamp(static_cast<int>(o.y * 2), min_height, mon_h - y);
    SetWindowSize(w, h);
  } else {
    ctx.config.signature = HermesConfig::V0;
    ctx.config.show_ui = true;
    ctx.config.show_help = true;
    ctx.config.show_grid = true;
    ctx.config.show_resources = true;
    ctx.config.show_attack_mode = true;
    ctx.config.show_visual_ranges = true;
    ctx.config.camera.zoom = 0.25f;
  }
  Hermes_SetupCamera(ctx);

  ctx.debug_mask = std::numeric_limits<uint64_t>::max();

  // One time initialization completed.
  ctx.initialized = true;
}

// Hermes Interface -----------------------------------------------------------

extern "C" {

Hermes* Hermes_Init() {
  return new Hermes();
}

void Hermes_Quit(Hermes* ctx) {
  if (ctx->initialized) {
    Hermes_Close(*ctx);
  }
  delete ctx;
}

void Hermes_Scene(Hermes* ctx, const MettaGrid* env) {
  assert(env != nullptr);
  ctx->next = env;  // Picked up in Hermes_Frame in the Hermes_Cache step.
}

bool Hermes_Frame(Hermes* ctx) {
  Hermes_Setup(*ctx);  // Engine (lazy one-time init)
  Hermes_Cache(*ctx);  // System (compute then reuse)
  Hermes_Input(*ctx);  // Player (user interactivity)
  Hermes_Batch(*ctx);  // Camera (broad phase bucket)
  Hermes_Image(*ctx);  // Render (narrow phase group)
  return ctx->initialized;
}

}  // extern "C"
#endif  // METTA_WITH_RAYLIB
