// extensions/report_resource_rewards.cpp
#include "extensions/report_resource_rewards.hpp"

#include "mettagrid_c.hpp"

void ReportResourceRewards::registerObservations(ObservationEncoder* enc) {
  _resource_rewards_feature = enc->register_feature("resource_rewards");
}

void ReportResourceRewards::onInit(const MettaGrid* env) {
  _num_agents = env->num_agents();
}

void ReportResourceRewards::onReset(MettaGrid* env) {
  // Add resource rewards to observations
  addResourceRewardsToObservations(env);
}

void ReportResourceRewards::onStep(MettaGrid* env) {
  // Add resource rewards to observations
  addResourceRewardsToObservations(env);
}

void ReportResourceRewards::addResourceRewardsToObservations(MettaGrid* env) {
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

ExtensionStats ReportResourceRewards::getStats() const {
  ExtensionStats stats;

  // Simple stats - just report that the extension is active
  stats["resource_rewards_reported"] = 1.0f;

  return stats;
}

REGISTER_EXTENSION("report_resource_rewards", ReportResourceRewards)
