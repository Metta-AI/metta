#ifndef EXTENSIONS_OBSERVE_EPISODE_COMPLETION_HPP_
#define EXTENSIONS_OBSERVE_EPISODE_COMPLETION_HPP_

#include <cmath>
#include <limits>

#include "extensions/mettagrid_extension.hpp"
#include "mettagrid_c.hpp"

class EpisodeCompletion : public MettaGridExtension {
public:
  void registerObservations(ObservationEncoder* enc) override {
    _episode_completion_pct_feature = enc->register_feature("episode_completion_pct");
  }

  void onInit(const MettaGrid* env, const GameConfig* /*config*/) override {
    _num_agents = env->num_agents();
    _max_steps = env->max_steps;
    _current_step = 0;
  }

  void onReset(MettaGrid* env) override {
    _current_step = 0;

    // Add initial completion (0%) to observations
    addEpisodeCompletionToObservations(env);
  }

  void onStep(MettaGrid* env) override {
    _current_step++;

    // Update completion percentage in observations
    addEpisodeCompletionToObservations(env);
  }

  std::string getName() const override {
    return "observe_episode_completion";
  }

private:
  ObservationType _episode_completion_pct_feature;
  size_t _max_steps;
  size_t _current_step;
  size_t _num_agents;

  void addEpisodeCompletionToObservations(MettaGrid* env) {
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
};

REGISTER_EXTENSION("observe_episode_completion", EpisodeCompletion)

#endif  // EXTENSIONS_OBSERVE_EPISODE_COMPLETION_HPP_
