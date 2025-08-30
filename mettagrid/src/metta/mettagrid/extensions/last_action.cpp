// extensions/episode_completion.cpp
#include <cmath>
#include <limits>

#include "extensions/episode_completion.hpp"
#include "mettagrid_c.hpp"

void EpisodeCompletion::registerObservations(ObservationEncoder* enc) {
  _episode_completion_pct_feature = enc->register_feature("episode_completion_pct");
}

void EpisodeCompletion::onInit(const MettaGrid* env) {
  _num_agents = env->num_agents();
  _max_steps = env->max_steps;
  _current_step = 0;
}

void EpisodeCompletion::onReset(MettaGrid* env) {
  _current_step = 0;

  // Add initial completion (0%) to observations
  addEpisodeCompletionToObservations(env);
}

void EpisodeCompletion::onStep(MettaGrid* env) {
  _current_step++;

  // Update completion percentage in observations
  addEpisodeCompletionToObservations(env);
}

void EpisodeCompletion::addEpisodeCompletionToObservations(MettaGrid* env) {
  // Calculate episode completion percentage
  ObservationType episode_completion_pct = 0;
  if (_max_steps > 0) {
    float fraction = (static_cast<float>(_current_step) / static_cast<float>(_max_steps));
    episode_completion_pct =
        static_cast<ObservationType>(std::round(fraction * std::numeric_limits<ObservationType>::max()));
  }

  // Create feature and value vectors for global observation
  std::vector<ObservationType> features = {_episode_completion_pct_feature};
  std::vector<ObservationType> values = {episode_completion_pct};

  // Write global observation for all agents
  for (size_t agent_idx = 0; agent_idx < _num_agents; agent_idx++) {
    writeGlobalObservations(env, agent_idx, features, values);
  }
}

ExtensionStats EpisodeCompletion::getStats() const {
  ExtensionStats stats;

  float completion_pct = 0.0f;
  if (_max_steps > 0) {
    completion_pct = (static_cast<float>(_current_step) / static_cast<float>(_max_steps)) * 100.0f;
  }

  stats["current_step"] = static_cast<float>(_current_step);
  stats["max_steps"] = static_cast<float>(_max_steps);
  stats["episode_completion_percentage"] = completion_pct;

  return stats;
}

REGISTER_EXTENSION("episode_completion", EpisodeCompletion)
