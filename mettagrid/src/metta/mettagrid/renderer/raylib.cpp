#include "raylib.hpp"

#include "grid.hpp"
#include "grid_object.hpp"
#include "mettagrid_c.hpp"

#include "objects/agent.hpp"
#include "objects/wall.hpp"

#include <raylib.h>

#include <format>
#include <string>
#include <unordered_map>
#include <vector>

#define METTA2D_DEBUG 1

#if METTA2D_DEBUG
# include <iostream> // TODO remove, for dumb debugging
# define METTA2D_DBG(fmt, ...) std::cout << std::format(fmt, __VA_ARGS__) << std::endl
#else
# define METTA2D_DBG(fmt, ...)
#endif

using namespace std::literals::string_view_literals;

// Constants ------------------------------------------------------------------

static constexpr Color FLOOR_COLOR = (Color){0xCF, 0xA9, 0x70, 0xFF};

static constexpr float SPRITE_SCALE = 256.0f;

// Structures -----------------------------------------------------------------

// Serializable user configuration.
struct RenderConfig {
    bool show_grid : 1;
};

// Resolved sprite names to rectangle indices in the sprite sheet.
struct RenderSprites {
    uint16_t agent[4]; // N, E, S, W
    uint16_t wall[9]; // 0-8 (N, NE, E, SE, S, SW, W, NW, C)
    std::vector<uint16_t> objects;
};

// Resolved type names to batch indices for specific render nodes.
struct RenderTypes {
    uint8_t agent;
    uint8_t wall;
};

// Lightweight data common to all render objects.
struct RenderNode {
    uint32_t x : 14;
    uint32_t y : 14;
    uint32_t a : 3;
    uint32_t b : 1;
};

struct RenderAgent {
    uint16_t id;
};

// Internal state, grouped by lifetimes.
struct Metta2D {
    // Local resources (always present)
    bool initialized : 1;
    RenderConfig config;
    Camera2D camera;

    // Asset resources (lazily loaded on first render)
    std::unordered_map<std::string, uint16_t> sprite_lookup;
    std::vector<Rectangle> sprites;
    RenderSprites sprite_atlas;
    Texture2D sprite_sheet;

    RenderTypes types;

    // Scene resources (updated on episode start)
    MettaGrid* self;
    MettaGrid* next;
    Grid* grid;

    std::vector<std::vector<RenderNode>> buckets;

    // Frame resources (updated on render)
    std::vector<RenderAgent> agents;
};

// Render Passes --------------------------------------------------------------

static inline Vector2 Position(RenderNode node) {
    return (Vector2){node.x * SPRITE_SCALE, node.y * SPRITE_SCALE};
}

static void DrawWalls(Metta2D& ctx) {
    auto& bucket = ctx.buckets[ctx.types.wall];
    for (auto node : bucket) {
        DrawTextureRec(ctx.sprite_sheet, ctx.sprites[ctx.sprite_atlas.wall[node.a]], Position(node), WHITE);
    }
}

static void DrawTrajectory(Metta2D& ctx) {
}

static void DrawObjects(Metta2D& ctx) {
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
        auto color = WHITE;
        DrawTextureRec(ctx.sprite_sheet, ctx.sprites[ctx.sprite_atlas.agent[node.a]], Position(node), color);
    }
}

static void DrawActions(Metta2D& ctx) {
}

static void DrawSelection(Metta2D& ctx) {
}

static void DrawInventory(Metta2D& ctx) {
}

static void DrawRewards(Metta2D& ctx) {
}

static void DrawVisibility(Metta2D& ctx) {
}

static void DrawGrid(Metta2D& ctx) {
}

static void DrawThoughtBubbles(Metta2D& ctx) {
}

static void DrawUI(Metta2D& ctx) {
    char buffer[256];
    snprintf(buffer, sizeof(buffer), "Step: %d, %d", ctx.self->current_step, (int32_t)(ctx.grid->objects.size() - 1));
    DrawText(buffer, 10, 10, 20, BLACK);
}

// Frame Rendering ------------------------------------------------------------

static void Metta2D_Image(Metta2D& ctx) {
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

static void Metta2D_Batch(Metta2D& ctx) {
    auto type_id = 0;
    for (auto& bucket : ctx.buckets) {
        if (type_id++ == ctx.types.wall) {
            continue;
        }

        bucket.clear();
    }

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

        // TODO camera visibility check
        auto loc = obj->location;

        ctx.buckets[obj->type_id].push_back({ .x = loc.r, .y = loc.c, .a = a });
    }
}

static void Metta2D_Cache(Metta2D& ctx) {
    if (ctx.self == ctx.next) {
        return;
    }
    ctx.self = ctx.next;
    ctx.grid = &ctx.self->grid();

    // TODO remove, for dumb debugging
    METTA2D_DBG("Env: {}x{}", ctx.self->obs_width, ctx.self->obs_height);
    METTA2D_DBG("Step: {}", ctx.self->current_step);
    METTA2D_DBG("Grid: {}x{}", ctx.grid->height, ctx.grid->width);
    for (auto& n : ctx.self->inventory_item_names) {
        METTA2D_DBG("Inventory: {}", n);
    }

    // Setup buckets for each object type.
    auto num_types = ctx.self->object_type_names.size();
    ctx.buckets.resize(num_types);
    ctx.buckets[ctx.types.wall].clear();

    // Map object type names to sprite indices.
    auto type_id = 0;
    ctx.sprite_atlas.objects.clear();
    ctx.sprite_atlas.objects.resize(num_types);
    for (auto& name : ctx.self->object_type_names) {
        auto sprite = 0;
        if (!name.empty()) {
            auto file = std::format("objects/{}.png", name); // TODO .color variants?
            sprite = ctx.sprite_lookup[file];

            /**/ if (name == "agent"sv) ctx.types.agent = type_id;
            else if (name == "wall"sv)  ctx.types.wall  = type_id;

            METTA2D_DBG("Object: {} = Sprite {}", name, ctx.sprite_lookup[file]);
        }
        ctx.sprite_atlas.objects[type_id++] = sprite;
    }

    METTA2D_DBG("Agent.type_id: {}", (int)ctx.types.agent);
    METTA2D_DBG("Wall.type_id: {}", (int)ctx.types.wall);

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
}

static void Metta2D_Setup(Metta2D& ctx) {
    if (ctx.initialized) {
        return;
    }

    InitWindow(800, 600, "MettaGrid");
    SetTargetFPS(60);

    ctx.sprite_sheet = LoadTexture("src/metta/mettagrid/renderer/atlas.png");
    SetTextureFilter(ctx.sprite_sheet, TEXTURE_FILTER_BILINEAR);
    SetTextureWrap(ctx.sprite_sheet, TEXTURE_WRAP_CLAMP);
    GenTextureMipmaps(&ctx.sprite_sheet);

    auto builtins = py::module::import("builtins");
    auto json = py::module::import("json");
    auto config = json.attr("load")(builtins.attr("open")("src/metta/mettagrid/renderer/atlas.json", "r"sv));

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

#if METTA2D_DEBUG
    std::vector<std::string> keys;
    keys.reserve(ctx.sprite_lookup.size());
    for (const auto& [key, value] : ctx.sprite_lookup) keys.push_back(key);
    std::sort(keys.begin(), keys.end());
    for (auto& k : keys) METTA2D_DBG("Sprite: {}", k);
#endif

    for (auto i = 0; i < 4; i++) {
        std::string_view orientation;
        switch (i) {
            case Orientation::Up:    orientation = "n"sv; break;
            case Orientation::Down:  orientation = "s"sv; break;
            case Orientation::Left:  orientation = "w"sv; break;
            case Orientation::Right: orientation = "e"sv; break;
        }
        auto file = std::format("agents/agent.{}.png", orientation);
        ctx.sprite_atlas.agent[i] = ctx.sprite_lookup[file];
    }

    for (auto i = 0; i < 9; i++) {
        std::string_view orientation;
        switch (i) {
            case 0: orientation = "n"sv; break;
            case 1: orientation = "e"sv; break;
            case 2: orientation = "s"sv; break;
            case 3: orientation = "w"sv; break;
            case 4: orientation = "we"sv; break;
            case 5: orientation = "ws"sv; break;
            case 6: orientation = "s"sv; break;
            case 7: orientation = "w"sv; break;
            case 8: orientation = "c"sv; break;
        }
        auto file = std::format("walls/wall.{}.png", orientation);
        ctx.sprite_atlas.wall[i] = ctx.sprite_lookup[file];
    }

    ctx.camera.offset = (Vector2){0, 0};
    ctx.camera.zoom = 0.1f;

    ctx.initialized = true;
}

// Metta2D Interface ----------------------------------------------------------

void Metta2D_Close(Metta2D* ctx) {
    if (ctx->initialized) {
        UnloadTexture(ctx->sprite_sheet);
        CloseWindow();

        ctx->initialized = false;
    }
}

void Metta2D_Scene(Metta2D* ctx, MettaGrid* env) {
    assert(env != nullptr);
    ctx->next = env;
}

void Metta2D_Frame(Metta2D* ctx) {
    Metta2D_Setup(*ctx); // Engine (lazy one-time init)
    Metta2D_Cache(*ctx); // System (compute then reuse)
    Metta2D_Batch(*ctx); // Camera (broad phase bucket)
    Metta2D_Image(*ctx); // Render (narrow phase group)
}

Metta2D* Metta2D_Init() {
    return new Metta2D();
}

void Metta2D_Quit(Metta2D* ctx) {
    Metta2D_Close(ctx);
    delete ctx;
}
