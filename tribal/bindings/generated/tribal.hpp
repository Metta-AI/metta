#ifndef INCLUDE_TRIBAL_H
#define INCLUDE_TRIBAL_H

#include <stdint.h>

#define MAP_AGENTS 15

#define OBSERVATION_WIDTH 11

#define OBSERVATION_HEIGHT 11

#define OBSERVATION_LAYERS 19

#define MAP_WIDTH 100

#define MAP_HEIGHT 50

struct SeqInt;

struct SeqFloat;

struct SeqBool;

struct TribalGameConfig;

struct TribalConfig;

struct TribalEnv;

struct SeqInt {

  private:

  uint64_t reference;

  public:

  void free();

};

struct SeqFloat {

  private:

  uint64_t reference;

  public:

  void free();

};

struct SeqBool {

  private:

  uint64_t reference;

  public:

  void free();

};

struct TribalGameConfig {
  int64_t max_steps;
  int64_t ore_per_battery;
  int64_t batteries_per_heart;
  bool enable_combat;
  double clippy_spawn_rate;
  int64_t clippy_damage;
  double heart_reward;
  double ore_reward;
  double battery_reward;
  double survival_penalty;
  double death_penalty;
};

struct TribalConfig {
  TribalGameConfig game;
  bool desync_episodes;
};

struct TribalEnv {

  private:

  uint64_t reference;

  public:

  TribalEnv(TribalConfig config);

  void free();

  /**
   * Reset the environment to initial state
   */
  void resetEnv();

  /**
   * Step environment with actions
   * actions: flat sequence of [action_type, argument] pairs
   * Length should be MapAgents * 2
   */
  bool step(SeqInt actions);

  /**
   * Get current observations as flat sequence
   */
  SeqInt getObservations();

  /**
   * Get current step rewards for each agent
   */
  SeqFloat getRewards();

  /**
   * Get terminated status for each agent
   */
  SeqBool getTerminated();

  /**
   * Get truncated status for each agent  
   */
  SeqBool getTruncated();

  /**
   * Get current step number
   */
  int64_t getCurrentStep();

  /**
   * Check if episode should end
   */
  bool isEpisodeDone();

  /**
   * Get text rendering of current state
   */
  const char* renderText();

};

extern "C" {

void tribal_seq_int_unref(SeqInt seq_int);

SeqInt tribal_new_seq_int();

int64_t tribal_seq_int_len(SeqInt seq_int);

int64_t tribal_seq_int_get(SeqInt seq_int, int64_t index);

void tribal_seq_int_set(SeqInt seq_int, int64_t index, int64_t value);

void tribal_seq_int_delete(SeqInt seq_int, int64_t index);

void tribal_seq_int_add(SeqInt seq_int, int64_t value);

void tribal_seq_int_clear(SeqInt seq_int);

void tribal_seq_float_unref(SeqFloat seq_float);

SeqFloat tribal_new_seq_float();

int64_t tribal_seq_float_len(SeqFloat seq_float);

double tribal_seq_float_get(SeqFloat seq_float, int64_t index);

void tribal_seq_float_set(SeqFloat seq_float, int64_t index, double value);

void tribal_seq_float_delete(SeqFloat seq_float, int64_t index);

void tribal_seq_float_add(SeqFloat seq_float, double value);

void tribal_seq_float_clear(SeqFloat seq_float);

void tribal_seq_bool_unref(SeqBool seq_bool);

SeqBool tribal_new_seq_bool();

int64_t tribal_seq_bool_len(SeqBool seq_bool);

bool tribal_seq_bool_get(SeqBool seq_bool, int64_t index);

void tribal_seq_bool_set(SeqBool seq_bool, int64_t index, bool value);

void tribal_seq_bool_delete(SeqBool seq_bool, int64_t index);

void tribal_seq_bool_add(SeqBool seq_bool, bool value);

void tribal_seq_bool_clear(SeqBool seq_bool);

bool tribal_check_error();

const char* tribal_take_error();

TribalGameConfig tribal_tribal_game_config(int64_t max_steps, int64_t ore_per_battery, int64_t batteries_per_heart, bool enable_combat, double clippy_spawn_rate, int64_t clippy_damage, double heart_reward, double ore_reward, double battery_reward, double survival_penalty, double death_penalty);

char tribal_tribal_game_config_eq(TribalGameConfig a, TribalGameConfig b);

TribalConfig tribal_tribal_config(TribalGameConfig game, bool desync_episodes);

char tribal_tribal_config_eq(TribalConfig a, TribalConfig b);

void tribal_tribal_env_unref(TribalEnv tribal_env);

TribalEnv tribal_new_tribal_env(TribalConfig config);

void tribal_tribal_env_reset_env(TribalEnv tribal);

bool tribal_tribal_env_step(TribalEnv tribal, SeqInt actions);

SeqInt tribal_tribal_env_get_observations(TribalEnv tribal);

SeqFloat tribal_tribal_env_get_rewards(TribalEnv tribal);

SeqBool tribal_tribal_env_get_terminated(TribalEnv tribal);

SeqBool tribal_tribal_env_get_truncated(TribalEnv tribal);

int64_t tribal_tribal_env_get_current_step(TribalEnv tribal);

bool tribal_tribal_env_is_episode_done(TribalEnv tribal);

const char* tribal_tribal_env_render_text(TribalEnv tribal);

int64_t tribal_default_max_steps();

TribalConfig tribal_default_tribal_config();

}

void SeqInt::free(){
  tribal_seq_int_unref(*this);
}

void SeqFloat::free(){
  tribal_seq_float_unref(*this);
}

void SeqBool::free(){
  tribal_seq_bool_unref(*this);
}

bool checkError() {
  return tribal_check_error();
};

const char* takeError() {
  return tribal_take_error();
};

TribalGameConfig tribalGameConfig(int64_t maxSteps, int64_t orePerBattery, int64_t batteriesPerHeart, bool enableCombat, double clippySpawnRate, int64_t clippyDamage, double heartReward, double oreReward, double batteryReward, double survivalPenalty, double deathPenalty) {
  return tribal_tribal_game_config(maxSteps, orePerBattery, batteriesPerHeart, enableCombat, clippySpawnRate, clippyDamage, heartReward, oreReward, batteryReward, survivalPenalty, deathPenalty);
};

TribalConfig tribalConfig(TribalGameConfig game, bool desyncEpisodes) {
  return tribal_tribal_config(game, desyncEpisodes);
};

TribalEnv::TribalEnv(TribalConfig config) {
  this->reference = tribal_new_tribal_env(config).reference;
}

void TribalEnv::free(){
  tribal_tribal_env_unref(*this);
}

void TribalEnv::resetEnv() {
  tribal_tribal_env_reset_env(*this);
};

bool TribalEnv::step(SeqInt actions) {
  return tribal_tribal_env_step(*this, actions);
};

SeqInt TribalEnv::getObservations() {
  return tribal_tribal_env_get_observations(*this);
};

SeqFloat TribalEnv::getRewards() {
  return tribal_tribal_env_get_rewards(*this);
};

SeqBool TribalEnv::getTerminated() {
  return tribal_tribal_env_get_terminated(*this);
};

SeqBool TribalEnv::getTruncated() {
  return tribal_tribal_env_get_truncated(*this);
};

int64_t TribalEnv::getCurrentStep() {
  return tribal_tribal_env_get_current_step(*this);
};

bool TribalEnv::isEpisodeDone() {
  return tribal_tribal_env_is_episode_done(*this);
};

const char* TribalEnv::renderText() {
  return tribal_tribal_env_render_text(*this);
};

int64_t defaultMaxSteps() {
  return tribal_default_max_steps();
};

TribalConfig defaultTribalConfig() {
  return tribal_default_tribal_config();
};

#endif
