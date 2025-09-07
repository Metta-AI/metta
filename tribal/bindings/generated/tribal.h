#ifndef INCLUDE_TRIBAL_H
#define INCLUDE_TRIBAL_H

#define MAP_AGENTS 15

#define OBSERVATION_WIDTH 11

#define OBSERVATION_HEIGHT 11

#define OBSERVATION_LAYERS 19

#define MAP_WIDTH 100

#define MAP_HEIGHT 50

typedef long long SeqInt;

typedef long long SeqFloat;

typedef long long SeqBool;

typedef struct TribalGameConfig {
  long long max_steps;
  long long ore_per_battery;
  long long batteries_per_heart;
  char enable_combat;
  double clippy_spawn_rate;
  long long clippy_damage;
  double heart_reward;
  double ore_reward;
  double battery_reward;
  double survival_penalty;
  double death_penalty;
} TribalGameConfig;

typedef struct TribalConfig {
  TribalGameConfig game;
  char desync_episodes;
} TribalConfig;

typedef long long TribalEnv;

void tribal_seq_int_unref(SeqInt seq_int);

SeqInt tribal_new_seq_int();

long long tribal_seq_int_len(SeqInt seq_int);

long long tribal_seq_int_get(SeqInt seq_int, long long index);

void tribal_seq_int_set(SeqInt seq_int, long long index, long long value);

void tribal_seq_int_delete(SeqInt seq_int, long long index);

void tribal_seq_int_add(SeqInt seq_int, long long value);

void tribal_seq_int_clear(SeqInt seq_int);

void tribal_seq_float_unref(SeqFloat seq_float);

SeqFloat tribal_new_seq_float();

long long tribal_seq_float_len(SeqFloat seq_float);

double tribal_seq_float_get(SeqFloat seq_float, long long index);

void tribal_seq_float_set(SeqFloat seq_float, long long index, double value);

void tribal_seq_float_delete(SeqFloat seq_float, long long index);

void tribal_seq_float_add(SeqFloat seq_float, double value);

void tribal_seq_float_clear(SeqFloat seq_float);

void tribal_seq_bool_unref(SeqBool seq_bool);

SeqBool tribal_new_seq_bool();

long long tribal_seq_bool_len(SeqBool seq_bool);

char tribal_seq_bool_get(SeqBool seq_bool, long long index);

void tribal_seq_bool_set(SeqBool seq_bool, long long index, char value);

void tribal_seq_bool_delete(SeqBool seq_bool, long long index);

void tribal_seq_bool_add(SeqBool seq_bool, char value);

void tribal_seq_bool_clear(SeqBool seq_bool);

char tribal_check_error();

char* tribal_take_error();

TribalGameConfig tribal_tribal_game_config(long long max_steps, long long ore_per_battery, long long batteries_per_heart, char enable_combat, double clippy_spawn_rate, long long clippy_damage, double heart_reward, double ore_reward, double battery_reward, double survival_penalty, double death_penalty);

char tribal_tribal_game_config_eq(TribalGameConfig a, TribalGameConfig b);

TribalConfig tribal_tribal_config(TribalGameConfig game, char desync_episodes);

char tribal_tribal_config_eq(TribalConfig a, TribalConfig b);

void tribal_tribal_env_unref(TribalEnv tribal_env);

TribalEnv tribal_new_tribal_env(TribalConfig config);

/**
 * Reset the environment to initial state
 */
void tribal_tribal_env_reset_env(TribalEnv tribal);

/**
 * Step environment with actions
 * actions: flat sequence of [action_type, argument] pairs
 * Length should be MapAgents * 2
 */
char tribal_tribal_env_step(TribalEnv tribal, SeqInt actions);

/**
 * Get current observations as flat sequence
 */
SeqInt tribal_tribal_env_get_observations(TribalEnv tribal);

/**
 * Get current step rewards for each agent
 */
SeqFloat tribal_tribal_env_get_rewards(TribalEnv tribal);

/**
 * Get terminated status for each agent
 */
SeqBool tribal_tribal_env_get_terminated(TribalEnv tribal);

/**
 * Get truncated status for each agent  
 */
SeqBool tribal_tribal_env_get_truncated(TribalEnv tribal);

/**
 * Get current step number
 */
long long tribal_tribal_env_get_current_step(TribalEnv tribal);

/**
 * Check if episode should end
 */
char tribal_tribal_env_is_episode_done(TribalEnv tribal);

/**
 * Get text rendering of current state
 */
char* tribal_tribal_env_render_text(TribalEnv tribal);

/**
 * Get default max steps value
 */
long long tribal_default_max_steps();

/**
 * Create default tribal configuration
 */
TribalConfig tribal_default_tribal_config();

#endif
