// extensions/resource_rewards.cpp
#include "extensions/resource_rewards.hpp"

#include <algorithm>

#include "mettagrid_c.hpp"
#include "objects/agent.hpp"

void ResourceRewards::registerObservations(ObservationEncoder* enc) {
  _resource_rewards_feature = enc->register_feature("resource_rewards");
}

void ResourceRewards::onInit(const MettaGrid* env) {
  _num_agents = env->num_agents();

  // Get number of inventory items from the environment
  // Assuming there's a method or property to get this
  _num_inventory_items = env->num_inventory_items();

  // Allocate storage for packed resource rewards
  _resource_rewards.resize(_num_agents, 0);
}

void ResourceRewards::onReset(MettaGrid* env) {
  // Initialize resource rewards
  updateResourceRewards(env);

  // Add initial resource rewards to observations
  addResourceRewardsToObservations(env);
}

void ResourceRewards::onStep(MettaGrid* env) {
  // Update resource rewards based on current agent states
  updateResourceRewards(env);

  // Add resource rewards to observations
  addResourceRewardsToObservations(env);
}

void ResourceRewards::updateResourceRewards(MettaGrid* env) {
  // Compute inventory rewards for each agent (1 bit per item, up to 8 items)
  // Each bit indicates whether the agent has a positive reward configured for that item type
  // This tells agents which items are worth collecting
  for (size_t agent_idx = 0; agent_idx < _num_agents; agent_idx++) {
    const Agent* agent = getAgent(env, agent_idx);
    uint8_t packed = 0;

    // Process up to 8 items (or all available items if fewer)
    size_t num_items = std::min(_num_inventory_items, size_t(8));

    for (size_t i = 0; i < num_items; i++) {
      // Check if this item has a reward configured
      auto item = static_cast<InventoryItem>(i);

      // Check if the agent has a positive reward for this item
      auto it = agent->resource_rewards.find(item);
      if (it != agent->resource_rewards.end() && it->second > 0) {
        // Set bit at position (7 - i) to 1
        // Item 0 goes to bit 7, item 1 to bit 6, etc.
        packed |= static_cast<uint8_t>(1 << (7 - i));
      }
    }

    _resource_rewards[agent_idx] = packed;
  }
}

void ResourceRewards::addResourceRewardsToObservations(MettaGrid* env) {
  // Write packed resource rewards as global observation for each agent
  for (size_t agent_idx = 0; agent_idx < _num_agents; agent_idx++) {
    // Create feature and value vectors for global observation
    std::vector<ObservationType> features = {_resource_rewards_feature};
    std::vector<ObservationType> values = {_resource_rewards[agent_idx]};

    // Write the observation
    writeGlobalObservations(env, agent_idx, features, values);
  }
}

ExtensionStats ResourceRewards::getStats() const {
  ExtensionStats stats;

  // Count how many agents have rewards for each item position
  std::vector<int> item_reward_counts(8, 0);

  for (size_t agent_idx = 0; agent_idx < _num_agents; agent_idx++) {
    uint8_t packed = _resource_rewards[agent_idx];

    // Check each bit
    for (int bit = 0; bit < 8; bit++) {
      if (packed & (1 << bit)) {
        // This agent has a reward for item at position (7 - bit)
        item_reward_counts[7 - bit]++;
      }
    }
  }

  // Report stats for each item slot
  for (int i = 0; i < 8; i++) {
    std::string key = "agents_with_reward_for_item_" + std::to_string(i);
    stats[key] = static_cast<float>(item_reward_counts[i]);
  }

  // Count total number of agents with any rewards
  int agents_with_any_reward = 0;
  for (size_t agent_idx = 0; agent_idx < _num_agents; agent_idx++) {
    if (_resource_rewards[agent_idx] != 0) {
      agents_with_any_reward++;
    }
  }
  stats["agents_with_any_reward"] = static_cast<float>(agents_with_any_reward);

  return stats;
}

REGISTER_EXTENSION("resource_rewards", ResourceRewards)
