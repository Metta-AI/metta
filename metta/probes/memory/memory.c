#include "memory.h"

int main() {
    Memory env = {.length = 16};
    env.observations = (float*)calloc(1, sizeof(unsigned char));
    env.actions = (int*)calloc(1, sizeof(int));
    env.rewards = (float*)calloc(1, sizeof(float));
    env.terminals = (unsigned char*)calloc(1, sizeof(unsigned char));

    c_reset(&env);
    c_render(&env);

}
