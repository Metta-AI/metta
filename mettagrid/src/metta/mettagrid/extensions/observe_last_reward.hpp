#ifndef EXTENSIONS_OBSERVE_LAST_REWARD_HPP_
#define EXTENSIONS_OBSERVE_LAST_REWARD_HPP_

#include <cmath>
#include <limits>
#include <vector>

#include "extensions/mettagrid_extension.hpp"
#include "mettagrid_c.hpp"

class LastReward : public MettaGridExtension {
public:
  void registerObservations(ObservationEncoder* enc) override {
    _last_reward_feature = enc->register_feature("last_reward");
  }

  void onInit(const MettaGrid* env, const GameConfig* /*config*/) override {
    _num_agents = env->num_agents();
    _last_rewards.resize(_num_agents, 0.0f);
  }

  void onReset(MettaGrid* env) override {
    // Reset all last rewards to 0
    std::fill(_last_rewards.begin(), _last_rewards.end(), 0.0f);

    // Add initial rewards (0) to observations
    addLastRewardToObservations(env);
  }

  void onStep(MettaGrid* env) override {
    // Get the current rewards from the environment
    auto rewards = getResourceRewards(env);

    // Update stored last rewards
    // Note: rewards are stored as uint8_t in the environment, need to convert back to float
    for (size_t agent_idx = 0; agent_idx < _num_agents; agent_idx++) {
      // Assuming rewards are normalized to [0, 255] range in the environment
      _last_rewards[agent_idx] = static_cast<float>(rewards[agent_idx]) / 255.0f;
    }

    // Update rewards in observations
    addLastRewardToObservations(env);
  }

  std::string getName() const override {
    return "observe_last_reward";
  }

private:
  ObservationType _last_reward_feature;
  size_t _num_agents;
  std::vector<float> _last_rewards;

  void addLastRewardToObservations(MettaGrid* env) {
    // Write last reward observation for each agent
    for (size_t agent_idx = 0; agent_idx < _num_agents; agent_idx++) {
      // Convert reward to observation format (multiply by 100 and round to int)
      ObservationType reward_int = static_cast<ObservationType>(std::round(_last_rewards[agent_idx] * 100.0f));

      // Create feature and value vectors for global observation
      std::vector<ObservationType> features = {_last_reward_feature};
      std::vector<ObservationType> values = {reward_int};

      // Write global observation for this agent
      writeGlobalObservations(env, agent_idx, features, values);
    }
  }
};

REGISTER_EXTENSION("observe_last_reward", LastReward)

#endif  // EXTENSIONS_OBSERVE_LAST_REWARD_HPP_
