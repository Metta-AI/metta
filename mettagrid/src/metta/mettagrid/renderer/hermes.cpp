#include "hermes.hpp"

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

#define METTA_HERMES_DEBUG 1

#if METTA_HERMES_DEBUG
# include <iostream>
# define METTA_HERMES_DBG(fmt, ...) std::cout << std::format(fmt ##sv, __VA_ARGS__) << std::endl
#else
# define METTA_HERMES_DBG(fmt, ...)
#endif

using namespace std::literals::string_view_literals;

// Constants ------------------------------------------------------------------

static constexpr Color FLOOR_COLOR = (Color){0xCF, 0xA9, 0x70, 0xFF};

static constexpr float SPRITE_SIZE = 200.0f;

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
struct RenderConfig {
    bool show_grid : 1;
};

// Resolved sprite names to rectangle indices in the sprite sheet.
struct RenderSprites {
    uint16_t agent[4]; // N, E, S, W
    uint16_t wall[17]; // WallTile enum values.
    std::vector<uint16_t> objects; // Indexed by type_id.
};

// Resolved type names to batch indices for specific render nodes.
struct RenderTypes {
    uint8_t agent;
    uint8_t wall;
};

// Lightweight data common to all render objects.
struct RenderNode {
    uint32_t x : 13;
    uint32_t y : 13;
    uint32_t a : 5; // Agent orientation or wall tile index.
    uint32_t b : 1; // Unused flag.
};

struct RenderAgent {
    uint16_t id;
};

// Internal state, grouped by lifetimes.
struct Hermes {
    // Local resources (always present)
    bool initialized : 1;
    RenderConfig config;
    Camera2D camera;

    // Asset resources (lazily loaded on first render)
    std::unordered_map<std::string, uint16_t> sprite_lookup;
    std::vector<Rectangle> sprites;
    RenderSprites sprite_atlas;
    Texture2D sprite_sheet;

    // Scene resources (updated on episode start)
    MettaGrid* next;
    MettaGrid* self;
    Grid* grid;

    RenderTypes types;
    std::vector<std::vector<RenderNode>> buckets;
    std::vector<RenderNode> wall_fills;

    // Frame resources (updated on render)
    std::vector<RenderAgent> agents;
};

// Render Passes --------------------------------------------------------------

static inline Vector2 Position(RenderNode node) {
    return (Vector2){node.x * SPRITE_SIZE, node.y * SPRITE_SIZE};
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
}

static void DrawObjects(Hermes& ctx) {
    auto type_id = 0;
    for (auto& bucket : ctx.buckets) {
        if (type_id != ctx.types.wall && type_id != ctx.types.agent) {
            auto sprite = ctx.sprites[ctx.sprite_atlas.objects[type_id]];
            for (auto node : bucket) {
                DrawTextureRec(ctx.sprite_sheet, sprite, Position(node), WHITE);
            }
        }
        type_id++;
    }

    auto node_id = 0;
    for (auto& node : ctx.buckets[ctx.types.agent]) {
        auto agent = ctx.agents[node_id++];
        Color color = {
            static_cast<uint8_t>(fmodf(agent.id * (float)M_PI, 1.0f) * 255),
            static_cast<uint8_t>(fmodf(agent.id * (float)M_E, 1.0f) * 255),
            static_cast<uint8_t>(fmodf(agent.id * (float)M_SQRT2, 1.0f) * 255),
            0xFF
        };
        DrawTextureRec(ctx.sprite_sheet, ctx.sprites[ctx.sprite_atlas.agent[node.a]], Position(node), color);
    }
}

static void DrawActions(Hermes& ctx) {
}

static void DrawSelection(Hermes& ctx) {
}

static void DrawInventory(Hermes& ctx) {
}

static void DrawRewards(Hermes& ctx) {
}

static void DrawVisibility(Hermes& ctx) {
}

static void DrawGrid(Hermes& ctx) {
}

static void DrawThoughtBubbles(Hermes& ctx) {
}

static void DrawUI(Hermes& ctx) {
    char buffer[256];
    snprintf(buffer, sizeof(buffer), "Step: %d, %d", ctx.self->current_step, (int32_t)(ctx.grid->objects.size() - 1));
    DrawText(buffer, 10, 10, 20, BLACK);
}

// Frame Rendering ------------------------------------------------------------

static void Hermes_Image(Hermes& ctx) {
    BeginDrawing();
        ClearBackground(FLOOR_COLOR);
        BeginMode2D(ctx.camera);
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
        EndMode2D();
        DrawUI(ctx);
    EndDrawing();
}

static void Hermes_Batch(Hermes& ctx) {
    // Clear all buckets, except for those containing static scene objects.
    auto type_id = 0;
    for (auto& bucket : ctx.buckets) {
        if (type_id++ == ctx.types.wall) {
            continue;
        }

        bucket.clear();
    }

    // Extract scene data from changing objects into buckets matching their type_id.
    auto& objects = ctx.grid->objects;
    for (auto it = objects.begin() + 1; it != objects.end(); it++) {
        auto obj = it->get();

        uint32_t a = 0;
        if (obj->type_id == ctx.types.wall) {
            continue; // Walls are batched once per episode.
        }
        else if (obj->type_id == ctx.types.agent) {
            auto agent = static_cast<Agent*>(obj);
            a = agent->orientation;
            ctx.agents.push_back({ .id = static_cast<uint16_t>(agent->id) });
        }

        // TODO camera visibility check (or capture all objects for replay frames)
        auto loc = obj->location;
        ctx.buckets[obj->type_id].push_back({ .x = loc.r, .y = loc.c, .a = a });
    }
}

static void Hermes_Input(Hermes& ctx) {
    // TODO should also constraint the camera to the grid bounds.
}

static void Hermes_Cache(Hermes& ctx) {
    // Skip unless the episode has changed, in which case we grab its scene data.
    if (ctx.self == ctx.next) {
        return;
    }
    ctx.self = ctx.next;
    ctx.grid = &ctx.self->grid();
    auto width = ctx.grid->width;
    auto height = ctx.grid->height;
    METTA_HERMES_DBG("Grid: {}x{}", width, height);

    // Map inventory item names to sprite indices.
    for (auto& n : ctx.self->inventory_item_names) {
        METTA_HERMES_DBG("Inventory: {}", n);
    }

    // Map object type names to sprite indices.
    auto num_types = ctx.self->object_type_names.size();
    ctx.sprite_atlas.objects.clear();
    ctx.sprite_atlas.objects.resize(num_types);

    auto type_id = 0;
    for (auto& name : ctx.self->object_type_names) {
        auto sprite = 0;
        if (!name.empty()) {
            auto file = std::format("objects/{}.png"sv, name); // TODO .color variants?
            sprite = ctx.sprite_lookup[file];

            /**/ if (name == "agent"sv) ctx.types.agent = type_id;
            else if (name == "wall"sv)  ctx.types.wall  = type_id;

            METTA_HERMES_DBG("Object: {} = Sprite {}", name, ctx.sprite_lookup[file]);
        }
        ctx.sprite_atlas.objects[type_id++] = sprite;
    }

    METTA_HERMES_DBG("Agent.type_id: {}", (int)ctx.types.agent);
    METTA_HERMES_DBG("Wall.type_id: {}", (int)ctx.types.wall);

    // Setup buckets for each object type.
    ctx.buckets.resize(num_types);
    ctx.buckets[ctx.types.wall].clear();
    ctx.wall_fills.clear();

    // Extract scene data from unchanging objects.
    auto& objects = ctx.grid->objects;
    auto& walls = ctx.buckets[ctx.types.wall];
    for (auto it = objects.begin() + 1; it != objects.end(); it++) {
        auto obj = it->get();
        if (obj->type_id == ctx.types.wall) { // TODO intermediate .a detection
            auto wall = static_cast<Wall*>(obj);
            auto loc = wall->location;
            walls.push_back({ .x = loc.r, .y = loc.c, .a = 0 });
        }
        /*else if (!obj->is_swappable()) { // TODO depends on is_swappable never changing
            auto loc = obj->location;
            ctx.buckets[obj->type_id].push_back({ .x = loc.r, .y = loc.c });
        }*/
    }

    // Construct a wall adjacency map and assign wall tile indices.
    auto wall_map_width = width + 2;
    auto wall_map_height = height + 2;
    std::vector<bool> wall_map(wall_map_width * wall_map_height);
    std::vector<bool> wall_del(walls.size());
    for (auto& wall : walls) {
        wall_map[(wall.y + 1) * wall_map_width + wall.x + 1] = true;
    }
    auto wall_idx = 0;
    for (auto& wall : walls) {
        auto x = wall.x + 1;
        auto y = wall.y + 1;

        int tile = WallTile_0;
        if (wall_map[(y + 1) * wall_map_width + x]) tile |= WallTile_S;
        if (wall_map[y * wall_map_width + x + 1]) tile |= WallTile_E;
        if (wall_map[(y - 1) * wall_map_width + x]) tile |= WallTile_N;
        if (wall_map[y * wall_map_width + x - 1]) tile |= WallTile_W;
        wall.a = tile;

        if ((tile & WallTile_SE) == WallTile_SE && wall_map[(y + 1) * wall_map_width + x + 1]) {
            ctx.wall_fills.push_back({ .x = wall.x, .y = wall.y });

            if ((tile & WallTile_NW) == WallTile_NW) {
                wall_del[wall_idx] = true;
            }
        }

        wall_idx++;
    }

    // Don't render walls covered by wall fills.
    auto write_idx = 0;
    for (auto read_idx = 0; read_idx < walls.size(); read_idx++) {
        if (!wall_del[read_idx]) {
            if (write_idx != read_idx) {
                walls[write_idx] = walls[read_idx];
            }
            write_idx++;
        }
    }
    walls.resize(write_idx);

    #if 0
    walls.clear();
    for (uint32_t i = 0; i < 17; i++) {
        walls.push_back({ .x = i, .y = 2, .a = i });
    }
    #endif

    // Center the camera and ensure it fits the grid.
    auto view_width = width * SPRITE_SIZE;
    auto view_height = height * SPRITE_SIZE;
    //ctx.camera.offset = (Vector2){ width / 2.0f, height / 2.0f };
    //ctx.camera.zoom = 0.1f;
}

static void Hermes_Setup(Hermes& ctx) {
    if (ctx.initialized) {
        return;
    }

    // Raylib initialization.
    InitWindow(1280, 720, "Lens of Hermes");
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
    for (const auto& [key, value] : ctx.sprite_lookup) keys.push_back(key);
    std::sort(keys.begin(), keys.end());
    for (auto& k : keys) METTA_HERMES_DBG("Sprite: {} ({}x{})", k,
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

    // Scene initialization.
    ctx.camera.offset = (Vector2){0, 0};
    ctx.camera.zoom = 0.1f;

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
        UnloadTexture(ctx->sprite_sheet);
        CloseWindow();
    }
    delete ctx;
}

void Hermes_Scene(Hermes* ctx, MettaGrid* env) {
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
