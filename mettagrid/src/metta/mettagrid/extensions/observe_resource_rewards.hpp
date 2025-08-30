#ifndef EXTENSIONS_OBSERVE_RESOURCE_REWARDS_HPP_
#define EXTENSIONS_OBSERVE_RESOURCE_REWARDS_HPP_

#include <vector>

#include "extensions/mettagrid_extension.hpp"
#include "mettagrid_c.hpp"

class ReportResourceRewards : public MettaGridExtension {
public:
  void registerObservations(ObservationEncoder* enc) override {
    _resource_rewards_feature = enc->register_feature("resource_rewards");
  }

  void onInit(const MettaGrid* env, const GameConfig* /*config*/) override {
    _num_agents = env->num_agents();
  }

  void onReset(MettaGrid* env) override {
    // Add resource rewards to observations
    addResourceRewardsToObservations(env);
  }

  void onStep(MettaGrid* env) override {
    // Add resource rewards to observations
    addResourceRewardsToObservations(env);
  }

  std::string getName() const override {
    return "observe_resource_rewards";
  }

private:
  ObservationType _resource_rewards_feature;
  size_t _num_agents;

  void addResourceRewardsToObservations(MettaGrid* env) {
    // Get the pre-calculated resource rewards for all agents
    auto resource_rewards = getResourceRewards(env);

    // For each agent, report their resource reward flags
    for (size_t agent_idx = 0; agent_idx < _num_agents; agent_idx++) {
      // Get the packed resource rewards byte for this agent
      uint8_t packed_rewards = resource_rewards[agent_idx];

      // Create feature and value vectors for global observation
      std::vector<ObservationType> features = {_resource_rewards_feature};
      std::vector<ObservationType> values = {packed_rewards};

      // Write the observation
      writeGlobalObservations(env, agent_idx, features, values);
    }
  }
};

REGISTER_EXTENSION("observe_resource_rewards", ReportResourceRewards)

#endif  // EXTENSIONS_OBSERVE_RESOURCE_REWARDS_HPP_
