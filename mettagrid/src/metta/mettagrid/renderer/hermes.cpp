#include "hermes.hpp"

#include "action_handler.hpp"
#include "grid.hpp"
#include "grid_object.hpp"
#include "mettagrid_c.hpp"

#include "objects/agent.hpp"
#include "objects/converter.hpp"
#include "objects/wall.hpp"

#include <raylib.h>

#include <format>
#include <string>
#include <unordered_map>
#include <vector>

#pragma clang diagnostic ignored "-Wgnu-anonymous-struct"
#pragma clang diagnostic ignored "-Wnested-anon-types"

#define METTA_HERMES_DEBUG 1

#if METTA_HERMES_DEBUG
# include <iostream>
# define METTA_HERMES_DBG(fmt, ...) std::cout << std::format(fmt ##sv, __VA_ARGS__) << std::endl
#else
# define METTA_HERMES_DBG(fmt, ...)
#endif

using namespace std::literals::string_view_literals;

// Constants ------------------------------------------------------------------

static constexpr float SPRITE_SIZE = 200.0f;
static constexpr float MOVE_SPEED = 10.0f;
static constexpr float ZOOM_SPEED = 2.0f;
static constexpr float DRAG_SPEED = 1.0f;
static constexpr float SCROLL_SPEED = 0.1f;

static constexpr Color FLOOR_COLOR = {0xCF, 0xA9, 0x70, 0xFF};
static constexpr Color OBJECT_COLORS[3] = {RED, GREEN, BLUE};

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

    WallTile_Fill = 16 // Special case for wall fill.
};

// Structures -----------------------------------------------------------------

// Serializable user configuration.
struct HermesConfig {
    Camera2D camera; // Also contains the window size in its offset parameter.
    GridObjectId selection; // NO_SELECTION for none, otherwise object index.

    union {
        uint8_t value;
        struct {
            bool show_grid : 1;
            bool show_resources : 1;
            bool show_attack_mode : 1;
            bool show_fog_of_war : 1;
            bool show_visual_ranges : 1;
            // Serialized as zeros.
            bool disable_file : 1;
            bool mouse_moved : 1;
            bool initialized : 1;
        };
    };
};

// Resolved sprite names to rectangle indices in the sprite sheet.
struct HermesSprites {
    struct Object {
        uint16_t base;
        uint16_t item;
        uint16_t color;
    };
    uint16_t agent[5]; // Orientation sprite indices, extra for frozen state.
    uint16_t wall[17]; // WallTile sprite indices.
    uint16_t grid;
    uint16_t reward;
    uint16_t selection;
    std::vector<uint16_t> actions;
    std::vector<Object> objects; // Indexed by type_id.
};

// Resolved type names to batch indices for specific render nodes.
struct HermesTypes {
    uint8_t agent;
    uint8_t wall;
};

// Lightweight data common to all render objects.
union HermesNode {
    struct {
        GridCoord x : 14;
        GridCoord y : 14;
        uint16_t a : 4; // Agent orientation/state, wall tile index, or color index.
    };
    struct {
        uint32_t _ : 28;
        uint32_t c : 2; // Converter color index.
        uint32_t o : 1; // Converter has output resources
        uint32_t _unused : 1;
    };
};

// HermesNode extension to the agent bucket.
struct HermesAgent {
    uint16_t color : 15;
    uint16_t frozen : 1;
};

// Internal state, grouped by lifetimes.
struct Hermes {
    // Local resources (always present)
    HermesConfig config;
    Vector2 last_mouse_pos;

    // Asset resources (lazily loaded on first render)
    std::unordered_map<std::string, uint16_t> sprite_lookup;
    std::vector<Rectangle> sprites;
    HermesSprites sprite_atlas;
    Texture2D sprite_sheet;

    // Scene resources (updated on episode start)
    const MettaGrid* next;
    const MettaGrid* self;
    const Grid* grid;

    HermesTypes types;
    std::vector<std::vector<HermesNode>> buckets; // Also contains frame nodes.
    std::vector<HermesNode> wall_fills;

    // Frame resources (updated on render)
    std::vector<HermesAgent> agents;
    std::vector<float> rewards;
    GridObject* selection; // Null if none or selected object invalid this frame.
    uint64_t items_mask; // Bitmask of buckets whose nodes carry an inventory.
    uint64_t color_mask; // Bitmask of buckets whose nodes carry a color index.
};

// Render Passes --------------------------------------------------------------

// HermesNode -> Raylib world vector, aligned to the grid.
static inline Vector2 Position(HermesNode node) {
    return {node.x * SPRITE_SIZE, node.y * SPRITE_SIZE};
}

static void DrawWalls(Hermes& ctx) {
    for (auto node : ctx.buckets[ctx.types.wall]) {
        DrawTextureRec(ctx.sprite_sheet, ctx.sprites[ctx.sprite_atlas.wall[node.a]], Position(node), WHITE);
    }

    auto fill_sprite = ctx.sprites[ctx.sprite_atlas.wall[WallTile_Fill]];
    for (auto node : ctx.wall_fills) {
        auto pos = Position(node);
        pos.x += SPRITE_SIZE / 2;
        pos.y += SPRITE_SIZE / 2 - 42;
        DrawTextureRec(ctx.sprite_sheet, fill_sprite, pos, WHITE);
    }
}

static void DrawTrajectory(Hermes& ctx) {
    if (ctx.selection == nullptr) {
        return;
    }

    // TODO capture frame data across time to draw this.
}

static void DrawObjects(Hermes& ctx) {
    auto type_id = 0u;
    for (const auto& bucket : ctx.buckets) {
        if (type_id != ctx.types.wall && type_id != ctx.types.agent) {
            auto sprite = ctx.sprite_atlas.objects[type_id];
            auto sprite_base = ctx.sprites[sprite.base];
            if (ctx.color_mask & (1 << type_id)) {
                for (auto node : bucket) {
                    auto sprite_item = ctx.sprites[sprite.item];
                    auto sprite_color = ctx.sprites[sprite.color];
                    DrawTextureRec(ctx.sprite_sheet, sprite_base, Position(node), WHITE);
                    DrawTextureRec(ctx.sprite_sheet, sprite_color, Position(node), OBJECT_COLORS[node.c]);
                    if (node.o) {
                        DrawTextureRec(ctx.sprite_sheet, sprite_item, Position(node), WHITE);
                    }
                }
            }
            else if (ctx.items_mask & (1 << type_id)) {
            }
            else {
                for (auto node : bucket) {
                    DrawTextureRec(ctx.sprite_sheet, sprite_base, Position(node), WHITE);
                }
            }
        }
        type_id++;
    }

    Color color;
    uint16_t sprite;
    auto node_id = 0u;
    for (auto node : ctx.buckets[ctx.types.agent]) {
        auto agent = ctx.agents[node_id++];
        if (agent.frozen) {
            sprite = 4;
            color = {0, 0, 0, 0xFF};
        }
        else {
            sprite = ctx.sprite_atlas.agent[node.a];
            color = {
                static_cast<uint8_t>(fmodf(agent.color * static_cast<float>(M_PI), 1.0f) * 0xFF),
                static_cast<uint8_t>(fmodf(agent.color * static_cast<float>(M_E), 1.0f) * 0xFF),
                static_cast<uint8_t>(fmodf(agent.color * static_cast<float>(M_SQRT2), 1.0f) * 0xFF),
                0xFF
            };
        }
        DrawTextureRec(ctx.sprite_sheet, ctx.sprites[sprite], Position(node), color);
    }
}

static void DrawActions(Hermes& ctx) {
    // Agent actions.

    // Converter actions.
}

static void DrawSelection(Hermes& ctx) {
    if (ctx.selection == nullptr) {
        return;
    }

    auto loc = ctx.selection->location;
    DrawTextureRec(ctx.sprite_sheet, ctx.sprites[ctx.sprite_atlas.selection], Position({.x = loc.r, .y = loc.c}), WHITE);
}

static void DrawInventory(Hermes& ctx) {
    if (!ctx.config.show_resources) {
        return;
    }

    // TODO
}

static void DrawRewards(Hermes& ctx) {
    constexpr auto REWARD_SIZE = SPRITE_SIZE / 8;
    auto node_id = 0u;
    auto sprite = ctx.sprites[ctx.sprite_atlas.reward];
    for (auto node : ctx.buckets[ctx.types.agent]) {
        auto reward = ctx.rewards[node_id++];
        auto width = std::min(32, static_cast<int32_t>(SPRITE_SIZE / reward));
        auto pos = Position(node);
        Rectangle dst = { pos.x, pos.y + SPRITE_SIZE - REWARD_SIZE, REWARD_SIZE, REWARD_SIZE };
        for (auto i = 0; i < reward; i++) {
            DrawTexturePro(ctx.sprite_sheet, sprite, dst, {0, 0}, 0, WHITE);
            dst.x += width;
        }
    }
}

static void DrawVisibility(Hermes& ctx) {
    if (!ctx.config.show_visual_ranges && !ctx.config.show_fog_of_war) {
        return;
    }

    size_t grid_width  = ctx.grid->width;
    size_t grid_height = ctx.grid->height;
    std::vector<bool> visibility_map(grid_width * grid_height);

    size_t obs_width  = ctx.self->obs_width;
    size_t obs_height = ctx.self->obs_height;
    auto view_width  = obs_width  / 2;
    auto view_height = obs_height / 2;

    auto update_visibility = [=, &visibility_map](size_t agent_x, size_t agent_y) {
        auto x = agent_x - view_width;
        auto y1 = agent_y - view_height;
        auto y2 = y1 + obs_width;
        for (; y1 < y2; y1++) {
            auto y = y1 * grid_width;
            auto x2 = x + obs_width;
            for (auto x1 = x; x1 < x2; x1++) {
                visibility_map[y + x1] = true;
            }
        }
    };

    if (ctx.selection != nullptr) {
        auto loc = ctx.selection->location;
        update_visibility(loc.r, loc.c);
    }
    else for (auto agent : ctx.buckets[ctx.types.agent]) {
        update_visibility(agent.x, agent.y);
    }

    Rectangle rect = {0, 0, SPRITE_SIZE, SPRITE_SIZE};
    Color color = {0, 0, 0, static_cast<uint8_t>(ctx.config.show_fog_of_war ? 0xFF : 0x20)};
    for (auto y = 0u; y < grid_height; y++) {
        rect.y = y * SPRITE_SIZE;
        for (auto x = 0u; x < grid_width; x++) {
            if (!visibility_map[y * grid_width + x]) {
                rect.x = x * SPRITE_SIZE;
                DrawRectangleRec(rect, color);
            }
        }
    }
}

static void DrawGrid(Hermes& ctx) {
    if (!ctx.config.show_grid) {
        return;
    }

    Vector2 pos;
    auto grid_width  = ctx.grid->width;
    auto grid_height = ctx.grid->height;
    auto grid_sprite = ctx.sprites[ctx.sprite_atlas.grid];
    for (auto y = 0; y < grid_height; y++) {
        pos.y = y * SPRITE_SIZE;
        for (auto x = 0; x < grid_width; x++) {
            pos.x = x * SPRITE_SIZE;
            DrawTextureRec(ctx.sprite_sheet, grid_sprite, pos, WHITE);
        }
    }
}

static void DrawThoughtBubbles(Hermes& ctx) {
    if (ctx.selection == nullptr || ctx.selection->type_id != ctx.types.agent) {
        return;
    }

    auto agent = static_cast<Agent*>(ctx.selection);
}

static void DrawAttackMode(Hermes& ctx) {
    if (!ctx.config.show_attack_mode) {
        return;
    }

    // TODO
}

static void DrawUI(Hermes& ctx) {
    char buffer[256];
    snprintf(buffer, sizeof(buffer), "Step: %d, %d", ctx.self->current_step, static_cast<int32_t>(ctx.grid->objects.size() - 1));
    DrawText(buffer, 10, 10, 20, BLACK);
}

// Frame Rendering ------------------------------------------------------------

static void Hermes_Image(Hermes& ctx) {
    BeginDrawing();
        ClearBackground(FLOOR_COLOR);
        BeginMode2D(ctx.config.camera);
            DrawWalls(ctx);
            DrawTrajectory(ctx);
            DrawObjects(ctx);
            DrawActions(ctx);
            DrawSelection(ctx);
            DrawInventory(ctx);
            DrawRewards(ctx);
            DrawVisibility(ctx);
            DrawGrid(ctx);
            DrawThoughtBubbles(ctx);
            DrawAttackMode(ctx);
        EndMode2D();
        DrawUI(ctx);
    EndDrawing();
}

// Breaks down the scene into independent buckets for each dynamic object type.
// This traverses the MettaGrid instance once per frame and simplifies drawing.
//
// Note that parts of the MettaGrid touched by single draw calls are not batched.
static void Hermes_Batch(Hermes& ctx) {
    // Clear all buckets, except for those containing static scene objects.
    auto type_id = 0;
    for (auto& bucket : ctx.buckets) {
        if (type_id++ == ctx.types.wall) {
            continue;
        }

        bucket.clear();
    }

    auto num_agents = ctx.self->num_agents();
    ctx.agents.clear(); // TODO agent IDs never change (also check if they're incremental, meaning == node_id)
    ctx.rewards.clear();
    ctx.agents.reserve(num_agents);
    ctx.rewards.reserve(num_agents);

    // Extract scene data from changing objects into buckets matching their type_id.
    for (const auto& ptr : ctx.grid->objects) {
        const auto obj = ptr.get();
        if (obj == nullptr) {
            continue;
        }

        // TODO camera visibility check (or capture all objects for replay frames)
        auto loc = obj->location;
        HermesNode node = { .x = loc.r, .y = loc.c, .a = 0 };

        auto t = obj->type_id;
        if (t == ctx.types.wall) {
            continue; // Walls are batched once per episode.
        }
        else if (t == ctx.types.agent) {
            const auto agent = static_cast<Agent*>(obj);
            node.a = agent->orientation;
            ctx.agents.push_back({
                .color = static_cast<uint16_t>(agent->agent_id),
                .frozen = agent->frozen > 0
            });
            ctx.rewards.push_back(*agent->reward);
        }
        else if (ctx.items_mask & (1u << t)) {
            // TODO separate inventory needed?
            //auto inv = static_cast<HasInventory*>(obj);
            if (ctx.color_mask & (1u << t)) {
                auto con = static_cast<Converter*>(obj);
                auto color = con->color;
                node.c = color > 2 ? 0 : color;
                node.o = con->output_resources.size() > 0;
            }
        }

        ctx.buckets[t].push_back(node);
    }

    // Update the selection pointer each frame, because the object may have been removed.
    ctx.selection = ctx.config.selection != NO_SELECTION
        ? ctx.grid->objects[ctx.config.selection].get()
        : nullptr;
}

static inline void Hermes_SetupCamera(Hermes& ctx) {
    ctx.config.camera.offset = {GetScreenWidth() / 2.0f, GetScreenHeight() / 2.0f};
}

static inline bool is_zero(float a) {
    return std::abs(a) < 0.001f;
}

// Handles user events and updates the Hermes configuration or camera in response.
static void Hermes_Input(Hermes& ctx) {
    // Configuration toggles.
    #define CONFIG(k, v) if (IsKeyPressed(k)) ctx.config.v = !ctx.config.v
    CONFIG(KEY_G, show_grid);
    CONFIG(KEY_R, show_resources);
    CONFIG(KEY_M, show_attack_mode);
    CONFIG(KEY_F, show_fog_of_war);
    CONFIG(KEY_V, show_visual_ranges);
    #undef CONFIG

    // Clearing selection.
    if (IsKeyPressed(KEY_ESCAPE)) {
        ctx.config.selection = NO_SELECTION;
    }

    // Cycling selection through agents.
    if (IsKeyPressed(KEY_SPACE)) {
        const auto& objects = ctx.grid->objects;
        auto num = objects.size();
        auto idx = ctx.config.selection == NO_SELECTION ? 0 : ctx.config.selection + 1;
        for (; idx < num; idx++) {
            const auto obj = objects[idx].get();
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
    if (IsMouseButtonDown(MOUSE_BUTTON_LEFT)) {
        auto delta_x = mouse_pos.x - ctx.last_mouse_pos.x;
        auto delta_y = mouse_pos.y - ctx.last_mouse_pos.y;
        if (!is_zero(delta_x) || !is_zero(delta_y)) {
            METTA_HERMES_DBG("Drag: {}x{}", delta_x, delta_y);
            ctx.config.camera.target.x -= delta_x / delta_scale * DRAG_SPEED;
            ctx.config.camera.target.y -= delta_y / delta_scale * DRAG_SPEED;
            ctx.config.mouse_moved = true;
        }
    }
    if (IsMouseButtonReleased(MOUSE_BUTTON_LEFT)) {
        if (!ctx.config.mouse_moved) {
            auto world_pos = GetScreenToWorld2D(mouse_pos, ctx.config.camera);
            auto grid_x = static_cast<GridCoord>(world_pos.x / SPRITE_SIZE);
            auto grid_y = static_cast<GridCoord>(world_pos.y / SPRITE_SIZE);
            Layer layer = GridLayer::GridLayerCount;
            do {
                auto obj = ctx.grid->object_at({grid_x, grid_y, --layer});
                if (obj != nullptr) {
                    METTA_HERMES_DBG("Pick: {}x{}.{} {}", grid_x, grid_y, layer, obj->id);
                    ctx.config.selection = obj->id;
                    break; // TODO cycle through layers if selected object is on this tile
                }
            } while (layer != 0);
        }
        ctx.config.mouse_moved = false;
    }
    ctx.last_mouse_pos = mouse_pos;

    // Keyboard camera controls.
    #define MOVE(k, d, dir) if (IsKeyDown(k)) ctx.config.camera.target.d += delta_time / delta_scale * dir * MOVE_SPEED * SPRITE_SIZE
    MOVE(KEY_W, y, -1);
    MOVE(KEY_S, y, +1);
    MOVE(KEY_A, x, -1);
    MOVE(KEY_D, x, +1);
    #undef MOVE

    #define ZOOM(k, dir) if (IsKeyDown(k)) zoom = delta_time * dir * ZOOM_SPEED
    ZOOM(KEY_Q, +1);
    ZOOM(KEY_E, -1);
    #undef ZOOM

    if (!is_zero(zoom)) {
        auto before = GetScreenToWorld2D(mouse_pos, ctx.config.camera);
        ctx.config.camera.zoom *= 1.0f + zoom;

        auto after = GetScreenToWorld2D(mouse_pos, ctx.config.camera);
        ctx.config.camera.target.x += before.x - after.x;
        ctx.config.camera.target.y += before.y - after.y;
    }

    // TODO should also constraint the camera to the grid bounds.

    // Window + camera = viewport.
    if (IsWindowResized()) {
        Hermes_SetupCamera(ctx);
    }
}

// Prepares the scene into a set of static resources reused for the entire episode.
static void Hermes_Cache(Hermes& ctx) {
    if (ctx.self == ctx.next) {
        return;
    }

    ctx.self = ctx.next;
    ctx.grid = &ctx.self->grid();
    ctx.config.selection = NO_SELECTION; // TODO not on first scene -> loaded config

    size_t grid_width = ctx.grid->width;
    size_t grid_height = ctx.grid->height;
    METTA_HERMES_DBG("Grid: {}x{}", grid_width, grid_height);

    // Map action names to sprite indices.
    const auto& action_handlers = ctx.self->action_handlers();
    ctx.sprite_atlas.actions.resize(action_handlers.size());

    auto action_id = 0u;
    for (const auto& handler : action_handlers) {
        const auto& name = handler->action_name();
        auto file = std::format("actions/icons/{}.png", name);
        ctx.sprite_atlas.actions[action_id++] = ctx.sprite_lookup[file];
        METTA_HERMES_DBG("Action: {} = Sprite {}", name, ctx.sprite_lookup[file]);
    }

    // Map inventory item names to sprite indices.
    for (const auto& n : ctx.self->inventory_item_names) {
        METTA_HERMES_DBG("Inventory: {}", n);
    }

    // Map object type names to sprite indices.
    const auto& type_names = ctx.self->object_type_names;
    auto num_types = type_names.size();
    ctx.sprite_atlas.objects.resize(num_types);

    uint8_t type_id = 0;
    for (const auto& name : type_names) {
        HermesSprites::Object sprite;
        auto found = false;
        if (!name.empty()) {
            std::string_view name_view = name;
            if (name_view.ends_with("_red"sv)) {
                name_view = name_view.substr(0, name_view.size() - 4);
            }
            else if (name_view.ends_with("_blue"sv)) {
                name_view = name_view.substr(0, name_view.size() - 5);
            }
            else if (name_view.ends_with("_green"sv)) {
                name_view = name_view.substr(0, name_view.size() - 6);
            }

            sprite.base  = ctx.sprite_lookup[std::format("objects/{}.png"sv, name_view)];
            sprite.item  = ctx.sprite_lookup[std::format("objects/{}.item.png"sv, name_view)];
            sprite.color = ctx.sprite_lookup[std::format("objects/{}.color.png"sv, name_view)];
            found = !(sprite.base == 0 || sprite.item == 0 || sprite.color == 0);
            METTA_HERMES_DBG("Object: {} = Sprite (base: {}, item: {}, color: {})", name_view, sprite.base, sprite.item, sprite.color);

            /**/ if (name == "agent"sv) ctx.types.agent = type_id;
            else if (name == "wall"sv)  ctx.types.wall  = type_id;
        }

        if (!found) {
            sprite = {};
        }

        ctx.sprite_atlas.objects[type_id++] = sprite;
    }

    METTA_HERMES_DBG("Agent.type_id: {}", static_cast<int>(ctx.types.agent));
    METTA_HERMES_DBG("Wall.type_id: {}", static_cast<int>(ctx.types.wall));

    // Setup buckets for each object type.
    ctx.buckets.resize(num_types);
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
        if (obj->type_id == ctx.types.wall) {
            auto loc = obj->location;
            walls.push_back({ .x = loc.r, .y = loc.c });
        }
        else if (auto inv = dynamic_cast<HasInventory*>(obj)) {
            items_mask |= 1 << obj->type_id;
            if (dynamic_cast<Converter*>(inv)) {
                color_mask |= 1 << obj->type_id;
            }
        }
        /*else if (!obj->is_swappable()) { // TODO depends on is_swappable never changing
            auto loc = obj->location;
            ctx.buckets[obj->type_id].push_back({ .x = loc.r, .y = loc.c });
        }*/
    }

    ctx.items_mask = items_mask;
    ctx.color_mask = color_mask;

    // Construct a wall adjacency map and assign each wall node a sprite index.
    auto wall_map_width  = grid_width  + 2; // Surround the grid with empty space to
    auto wall_map_height = grid_height + 2; // avoid testing for borders at the edges.
    std::vector<bool> wall_map(wall_map_width * wall_map_height);
    std::vector<bool> wall_del(walls.size()); // Mark wall nodes covered by wall fills.
    for (auto wall : walls) {
        wall_map[(wall.y + 1) * wall_map_width + wall.x + 1] = true;
    }
    auto wall_idx = 0u;
    for (auto& wall : walls) {
        auto x = wall.x + 1u;
        auto y = wall.y + 1u;

        #define WALL(x, y) wall_map[(y) * wall_map_width + (x)]
        WallTile tile = WallTile_0;
        if (WALL(x, y + 1)) tile |= WallTile_S;
        if (WALL(x + 1, y)) tile |= WallTile_E;
        if (WALL(x, y - 1)) tile |= WallTile_N;
        if (WALL(x - 1, y)) tile |= WallTile_W;
        wall.a = tile;

        if ((tile & WallTile_SE) == WallTile_SE && WALL(x + 1, y + 1)) {
            ctx.wall_fills.push_back({ .x = wall.x, .y = wall.y });

            if ((tile & WallTile_NW) == WallTile_NW && WALL(x + 1, y - 1) &&WALL(x - 1, y - 1) && WALL(x - 1, y + 1)) {
                wall_del[wall_idx] = true;
            }
        }

        wall_idx++;
        #undef WALL
    }

    // Don't render walls covered by wall fills.
    auto write_idx = 0u;
    for (size_t read_idx = 0; read_idx < walls.size(); read_idx++) {
        if (!wall_del[read_idx]) {
            if (write_idx != read_idx) {
                walls[write_idx] = walls[read_idx];
            }
            write_idx++;
        }
    }
    walls.resize(write_idx);

    #if 0 // Debug draw all wall tiles
    walls.clear(); ctx.wall_fills.clear();
    for (auto i = 0; i < 17; i++) {
        walls.push_back({ .x = i * 2, .y = 2, .a = i });
    }
    #endif

    // Center the camera and ensure it fits the grid.
    auto view_width = grid_width * SPRITE_SIZE;
    auto view_height = grid_height * SPRITE_SIZE;
    //ctx.camera.target = (Vector2){ width / 2.0f, height / 2.0f };
    //ctx.camera.zoom = 0.1f;
}

// Initializes the Hermes instance, its Raylib window, and sprite sheet.
static void Hermes_Setup(Hermes& ctx) {
    if (ctx.config.initialized) {
        return;
    }

    // TODO load config

    // Raylib initialization.
    SetConfigFlags(FLAG_WINDOW_RESIZABLE);
    InitWindow(1280, 960, "Lens of Hermes");
    SetTargetFPS(60);

    // Sprite sheet texture.
    ctx.sprite_sheet = LoadTexture("mettagrid/src/metta/mettagrid/renderer/atlas.png");
    SetTextureFilter(ctx.sprite_sheet, TEXTURE_FILTER_BILINEAR);
    SetTextureWrap(ctx.sprite_sheet, TEXTURE_WRAP_CLAMP);
    GenTextureMipmaps(&ctx.sprite_sheet);

    // Sprite sheet atlas and lookup.
    auto builtins = py::module::import("builtins");
    auto json = py::module::import("json");
    auto config = json.attr("load")(builtins.attr("open")("mettagrid/src/metta/mettagrid/renderer/atlas.json", "r"sv));

    auto num_sprites = py::len(config);
    ctx.sprite_lookup.reserve(num_sprites);
    ctx.sprites.reserve(num_sprites);

    for (auto& [key, value] : config.cast<py::dict>()) {
        auto name = key.cast<std::string>();
        auto sprite = value.cast<py::list>();
        ctx.sprite_lookup[name] = static_cast<uint16_t>(ctx.sprites.size());
        ctx.sprites.emplace_back(Rectangle{
            .x = sprite[0].cast<float>(),
            .y = sprite[1].cast<float>(),
            .width = sprite[2].cast<float>(),
            .height = sprite[3].cast<float>()
        });
    }

#if METTA_HERMES_DEBUG
    std::vector<std::string> keys;
    keys.reserve(ctx.sprite_lookup.size());
    for (const auto& kv : ctx.sprite_lookup) keys.push_back(kv.first);
    std::sort(keys.begin(), keys.end());
    for (const auto& k : keys) METTA_HERMES_DBG("Sprite: {} ({}x{})", k,
        ctx.sprites[ctx.sprite_lookup[k]].width,
        ctx.sprites[ctx.sprite_lookup[k]].height);
#endif

    // Predefined sprites.
    auto agent_sprite = [&ctx](std::string_view suffix, Orientation orientation) {
        auto file = std::format("agents/agent.{}.png"sv, suffix);
        ctx.sprite_atlas.agent[orientation] = ctx.sprite_lookup[file];
    };

    auto wall_sprite = [&ctx](std::string_view suffix, WallTile tile) {
        auto file = std::format("objects/wall.{}.png"sv, suffix);
        ctx.sprite_atlas.wall[tile] = ctx.sprite_lookup[file];
    };

    agent_sprite("n", Orientation::Up);
    agent_sprite("s", Orientation::Down);
    agent_sprite("w", Orientation::Left);
    agent_sprite("e", Orientation::Right);
    ctx.sprite_atlas.agent[4] = ctx.sprite_lookup["agents/frozen.png"];

    wall_sprite("0", WallTile_0); // 0
    wall_sprite("e", WallTile_E); // 1
    wall_sprite("s", WallTile_S); // 2
    wall_sprite("se", WallTile_S | WallTile_E);
    wall_sprite("w", WallTile_W); // 3
    wall_sprite("we", WallTile_W | WallTile_E);
    wall_sprite("ws", WallTile_W | WallTile_S);
    wall_sprite("wse", WallTile_W | WallTile_S | WallTile_E);
    wall_sprite("n", WallTile_N); // 4
    wall_sprite("ne", WallTile_N | WallTile_E);
    wall_sprite("ns", WallTile_N | WallTile_S);
    wall_sprite("nse", WallTile_N | WallTile_S | WallTile_E);
    wall_sprite("nw", WallTile_N | WallTile_W);
    wall_sprite("nwe", WallTile_N | WallTile_W | WallTile_E);
    wall_sprite("nws", WallTile_N | WallTile_W | WallTile_S);
    wall_sprite("nwse", WallTile_N | WallTile_W | WallTile_S | WallTile_E);
    wall_sprite("fill", WallTile_Fill);

    ctx.sprite_atlas.grid = ctx.sprite_lookup["objects/grid.png"];
    ctx.sprite_atlas.reward = ctx.sprite_lookup["resources/reward.png"];
    ctx.sprite_atlas.selection = ctx.sprite_lookup["selection.png"];

    // Scene initialization.

    // TODO defaults if no config file found
    Hermes_SetupCamera(ctx);
    ctx.config.camera.zoom = 0.25f;
    ctx.config.show_grid = true;
    ctx.config.show_resources = true;
    ctx.config.show_attack_mode = true;

    // One time initialization completed.
    ctx.config.initialized = true;
}

// Hermes Interface -----------------------------------------------------------

extern "C" {

Hermes* Hermes_Init() {
    return new Hermes();
}

void Hermes_Quit(Hermes* ctx) {
    if (ctx->config.initialized) {
        UnloadTexture(ctx->sprite_sheet);
        CloseWindow();

        // TODO save config
    }
    delete ctx;
}

void Hermes_Scene(Hermes* ctx, const MettaGrid* env) {
    assert(env != nullptr);
    ctx->next = env; // Picked up in Hermes_Frame in the Hermes_Cache step.
}

void Hermes_Frame(Hermes* ctx) {
    Hermes_Setup(*ctx); // Engine (lazy one-time init)
    Hermes_Cache(*ctx); // System (compute then reuse)
    Hermes_Input(*ctx); // Player (user interactivity)
    Hermes_Batch(*ctx); // Camera (broad phase bucket)
    Hermes_Image(*ctx); // Render (narrow phase group)
}

} // extern "C"
