#include <stdlib.h>
#include <string.h>
#include "raylib.h"

const Color PUFF_RED = (Color){187, 0, 0, 255};
const Color PUFF_CYAN = (Color){0, 187, 187, 255};
const Color PUFF_WHITE = (Color){241, 241, 241, 241};
const Color PUFF_BACKGROUND = (Color){6, 24, 24, 255};

// Only use floats!
typedef struct {
    float score;
    float n; // Required as the last field
} Log;

typedef struct {
    Log log;
    float* observations;
    int* actions;
    float* rewards;
    unsigned char* terminals;
    int length;
    int goal;
    int tick;
} Memory;

void c_reset(Memory* env) {
    // resets goal to -1 or 1
    // resets observations[0] to goal
    // resets tick to 0
    env->goal = (rand()%2 == 0) ? -1 : 1;
    env->observations[0] = env->goal;
    env->tick = 0;
}

void c_step(Memory* env){
    env->rewards[0] = 0;
    env->terminals[0] = 0;
    env->observations[0] = 0;
    env->tick++;

    if (env->tick < env->length) {
        return;
    }

    float val = 0.0f
    if (env->actions[0] == 0 && env->goal == -1) {
        val = 1.0f;
    } else if (env->actions[0] == 1 && env->goal == 1) {
        val = 1.0f;
    }

    c_reset(env);
    env->rewards[0] = val;
    env->terminals[0] = 1;
    env->log.score += val;
    env->log.n += 1;
}

void c_render(Memory* env) {
    if (!IsWindowReady()) {
        InitWindow(960, 480, "Probe Memory");
        SetTargetFPS(5);
    }

    if (IsKeyDown(KEY_SPACE)) {
        exit(0);
    }

    BeginDrawing();
    DrawRectangle(0, 0, 480, 480, (env->goal == -1 ? PUFF_CYAN : PUFF_RED));
    DrawRectangle(480, 0, 480, 480, (env->rewards[0] == 0 ? PUFF_RED: GREEN));
    DrawText(TextFormat("Tick %.0d. Simon says...", env->tick), 20, 20, 20, PUFF_WHITE);

    ClearBackground(PUFF_BACKGROUND);
    EndDrawing();
}

void c_close(Memory* env) {
    if (IsWindowReady()) {
        CloseWindow();
    }
}


