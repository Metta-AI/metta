#ifndef OBJECTS_AGENT_HPP_
#define OBJECTS_AGENT_HPP_

#include <algorithm>
#include <array>
#include <cassert>
#include <string>
#include <vector>
#include <random>

#include "../event.hpp"
#include "../stats_tracker.hpp"
#include "agent_config.hpp"
#include "constants.hpp"
#include "objects/box.hpp"
#include "types.hpp"

class Agent : public GridObject {
public:
  ObservationType group;
  short frozen;
  short freeze_duration;
  Orientation orientation;
  // inventory is a map of item to amount.
  // keys should be deleted when the amount is 0, to keep iteration faster.
  // however, this should not be relied on for correctness.
  std::map<InventoryItem, InventoryQuantity> inventory;
  std::map<InventoryItem, RewardType> resource_rewards;
  std::map<InventoryItem, RewardType> resource_reward_max;
  std::map<std::string, RewardType> stat_rewards;
  std::map<std::string, RewardType> stat_reward_max;
  std::map<InventoryItem, InventoryQuantity> resource_limits;
  float action_failure_penalty;
  std::string group_name;
  ObservationType color;
  ObservationType glyph;
  // Despite being a GridObjectId, this is different from the `id` property.
  // This is the index into MettaGrid._agents (std::vector<Agent*>)
  GridObjectId agent_id;
  StatsTracker stats;
  RewardType current_stat_reward;
  RewardType* reward;
  // Visitation count grid: tracks how many times the agent has visited each position
  std::vector<std::vector<unsigned int>> visitation_grid;
  bool visitation_counts_enabled = false;
  GridLocation prev_location;
  std::string prev_action_name;
  unsigned int steps_without_motion;
  Box* box;
  std::map<InventoryItem, float> resource_loss_prob;
  EventManager* event_manager;
  std::mt19937* rng;  // Pointer to the MettaGrid's RNG

  // Resource identity tracking
  struct ResourceInstance {
    uint64_t id;  // Unique identifier for this resource instance
    InventoryItem item_type;
    unsigned int creation_timestep;
  };
  std::map<uint64_t, ResourceInstance> resource_instances;  // Map from resource ID to instance
  std::map<InventoryItem, std::vector<uint64_t>> item_to_resources;  // Map from item type to resource IDs
  uint64_t next_resource_id;


  Agent(GridCoord r, GridCoord c, const AgentConfig& config)
      : group(config.group_id),
        frozen(0),
        freeze_duration(config.freeze_duration),
        orientation(Orientation::North),
        inventory(),
        resource_rewards(config.resource_rewards),
        resource_reward_max(config.resource_reward_max),
        stat_rewards(config.stat_rewards),
        stat_reward_max(config.stat_reward_max),
        resource_limits(config.resource_limits),
        action_failure_penalty(config.action_failure_penalty),
        group_name(config.group_name),
        color(0),
        glyph(0),
        agent_id(0),
        stats(),  // default constructor
        current_stat_reward(0),
        reward(nullptr),
        prev_location(r, c, GridLayer::AgentLayer),
        prev_action_name(""),
        steps_without_motion(0),
        box(nullptr),
        resource_loss_prob(config.resource_loss_prob),
        event_manager(nullptr),
        rng(nullptr),
        next_resource_id(1) {
    populate_initial_inventory(config.initial_inventory);
    GridObject::init(config.type_id, config.type_name, GridLocation(r, c, GridLayer::AgentLayer));
  }

  void init(RewardType* reward_ptr) {
    this->reward = reward_ptr;
  }

  void set_event_manager(EventManager* event_manager_ptr) {
    this->event_manager = event_manager_ptr;
  }

  void set_rng(std::mt19937* rng_ptr) {
    this->rng = rng_ptr;
  }

  // Resource instance management
  uint64_t create_resource_instance(InventoryItem item_type, unsigned int current_timestep) {
    uint64_t resource_id = next_resource_id++;
    resource_instances[resource_id] = {resource_id, item_type, current_timestep};
    item_to_resources[item_type].push_back(resource_id);
    return resource_id;
  }

  void remove_resource_instance(uint64_t resource_id) {
    auto it = resource_instances.find(resource_id);
    if (it != resource_instances.end()) {
      InventoryItem item_type = it->second.item_type;
      resource_instances.erase(it);

      // Remove from item_to_resources
      auto& resources = item_to_resources[item_type];
      resources.erase(std::remove(resources.begin(), resources.end(), resource_id), resources.end());
      if (resources.empty()) {
        item_to_resources.erase(item_type);
      }
    }
  }

  uint64_t get_random_resource_id(InventoryItem item_type) {
    auto it = item_to_resources.find(item_type);
    if (it == item_to_resources.end() || it->second.empty()) {
      return 0;  // No resources of this type
    }

    // Use proper RNG if available, otherwise fall back to rand()
    if (this->rng) {
      std::uniform_int_distribution<size_t> dist(0, it->second.size() - 1);
      size_t index = dist(*this->rng);
      return it->second[index];
    } else {
      size_t index = rand() % it->second.size();
      return it->second[index];
    }
  }

  // Helper method to create resource instances and schedule loss events
  void create_and_schedule_resources(InventoryItem item_type, int count, unsigned int current_timestep) {
    if (!this->event_manager || !this->rng) {
      return;
    }

    // Check if this item type has a loss probability
    auto prob_it = this->resource_loss_prob.find(item_type);
    if (prob_it == this->resource_loss_prob.end() || prob_it->second <= 0.0f) {
      // No loss probability, just create resource instances without scheduling events
      for (int i = 0; i < count; i++) {
        create_resource_instance(item_type, current_timestep);
      }
      return;
    }

    // Create resource instances and schedule loss events
    for (int i = 0; i < count; i++) {
      uint64_t resource_id = create_resource_instance(item_type, current_timestep);

      // Use exponential distribution to determine lifetime
      std::exponential_distribution<float> exp_dist(prob_it->second);
      float lifetime = exp_dist(*this->rng);
      unsigned int loss_timestep = current_timestep + static_cast<unsigned int>(std::ceil(lifetime));

      // Schedule the loss event with the resource ID as the argument
      this->event_manager->schedule_event(EventType::StochasticResourceLoss,
                                        loss_timestep - current_timestep,
                                        this->id,
                                        static_cast<EventArg>(resource_id));
    }
  }

  void populate_initial_inventory(const std::map<InventoryItem, InventoryQuantity>& initial_inventory) {
    for (const auto& [item, amount] : initial_inventory) {
      if (amount > 0) {
        this->inventory[item] = amount;

        // Create resource instances for initial inventory
        if (this->event_manager && this->rng) {
          unsigned int current_timestep = this->event_manager->get_current_timestep();
          create_and_schedule_resources(item, amount, current_timestep);
        }
      }
    }
  }

  void init_visitation_grid(GridCoord height, GridCoord width) {
    visitation_grid.resize(height, std::vector<unsigned int>(width, 0));
    visitation_counts_enabled = true;
  }

  void reset_visitation_counts() {
    for (auto& row : visitation_grid) {
      std::fill(row.begin(), row.end(), 0);
    }
  }

  void increment_visitation_count(GridCoord r, GridCoord c) {
    if (!visitation_counts_enabled) return;

    if (r < static_cast<GridCoord>(visitation_grid.size()) && c < static_cast<GridCoord>(visitation_grid[0].size())) {
      visitation_grid[r][c]++;
    }
  }

  std::array<unsigned int, 5> get_visitation_counts() const {
    std::array<unsigned int, 5> counts = {0, 0, 0, 0, 0};
    if (!visitation_grid.empty()) {
      counts[0] = get_visitation_count(location.r, location.c);  // center

      // Handle potential underflow at map edge
      if (location.r > 0) {
        counts[1] = get_visitation_count(location.r - 1, location.c);  // up
      }
      counts[2] = get_visitation_count(location.r + 1, location.c);  // down

      if (location.c > 0) {
        counts[3] = get_visitation_count(location.r, location.c - 1);  // left
      }
      counts[4] = get_visitation_count(location.r, location.c + 1);  // right
    }
    return counts;
  }

  InventoryDelta update_inventory(InventoryItem item, InventoryDelta attempted_delta) {
    // Get the initial amount (0 if item doesn't exist)
    InventoryQuantity initial_amount = 0;
    auto inv_it = this->inventory.find(item);
    if (inv_it != this->inventory.end()) {
      initial_amount = inv_it->second;
    }

    // Calculate the new amount with clamping
    InventoryQuantity new_amount = static_cast<InventoryQuantity>(std::clamp(
        static_cast<int>(initial_amount + attempted_delta), 0, static_cast<int>(this->resource_limits[item])));

    InventoryDelta delta = new_amount - initial_amount;

    // Handle inventory changes
    if (delta > 0) {
      // Adding inventory - create resource instances and schedule loss events
      this->stats.add(this->stats.resource_name(item) + ".gained", delta);

      if (this->event_manager && this->agent_id != 0 && this->rng) {
        unsigned int current_timestep = this->event_manager->get_current_timestep();
        create_and_schedule_resources(item, delta, current_timestep);
      }

      // Update inventory
      this->inventory[item] = new_amount;
    } else if (delta < 0) {
      // Removing inventory - remove random resource instances
      this->stats.add(this->stats.resource_name(item) + ".lost", -delta);

      // Remove random resource instances
      for (int i = 0; i < -delta; i++) {
        uint64_t resource_id = get_random_resource_id(item);
        if (resource_id != 0) {
          remove_resource_instance(resource_id);
        }
      }

      // Update inventory
      if (new_amount > 0) {
        this->inventory[item] = new_amount;
      } else {
        this->inventory.erase(item);
      }
    }

    // Update resource rewards incrementally
    this->_update_resource_reward(item, initial_amount, new_amount);

    return delta;
  }

  void compute_stat_rewards() {
    if (this->stat_rewards.empty()) {
      return;
    }

    float new_stat_reward = 0;
    auto stat_dict = this->stats.to_dict();

    for (const auto& [stat_name, reward_per_unit] : this->stat_rewards) {
      if (stat_dict.count(stat_name) > 0) {
        float stat_value = stat_dict[stat_name];

        float stats_reward = stat_value * reward_per_unit;
        if (this->stat_reward_max.count(stat_name) > 0) {
          stats_reward = std::min(stats_reward, this->stat_reward_max.at(stat_name));
        }

        new_stat_reward += stats_reward;
      }
    }

    // Update the agent's reward with the difference
    float reward_delta = new_stat_reward - this->current_stat_reward;
    if (reward_delta != 0.0f) {
      *this->reward += reward_delta;
      this->current_stat_reward = new_stat_reward;
    }
  }

  bool swappable() const override {
    return this->frozen;
  }

  std::vector<PartialObservationToken> obs_features() const override {
    const size_t num_tokens = this->inventory.size() + 5 + (glyph > 0 ? 1 : 0);

    std::vector<PartialObservationToken> features;
    features.reserve(num_tokens);

    features.push_back({ObservationFeature::TypeId, static_cast<ObservationType>(type_id)});
    features.push_back({ObservationFeature::Group, static_cast<ObservationType>(group)});
    features.push_back({ObservationFeature::Frozen, static_cast<ObservationType>(frozen != 0 ? 1 : 0)});
    features.push_back({ObservationFeature::Orientation, static_cast<ObservationType>(orientation)});
    features.push_back({ObservationFeature::Color, static_cast<ObservationType>(color)});
    if (glyph != 0) features.push_back({ObservationFeature::Glyph, static_cast<ObservationType>(glyph)});

    for (const auto& [item, amount] : this->inventory) {
      // inventory should only contain non-zero amounts
      assert(amount > 0);
      auto item_observation_feature = static_cast<ObservationType>(InventoryFeatureOffset + item);
      features.push_back({item_observation_feature, static_cast<ObservationType>(amount)});
    }

    return features;
  }

private:
  inline void _update_resource_reward(InventoryItem item, InventoryQuantity old_amount, InventoryQuantity new_amount) {
    // Early exit if this item doesn't contribute to rewards
    auto reward_it = this->resource_rewards.find(item);
    if (reward_it == this->resource_rewards.end()) {
      return;
    }

    // Calculate the old and new contributions from this item
    float reward_per_item = reward_it->second;
    float old_contribution = reward_per_item * old_amount;
    float new_contribution = reward_per_item * new_amount;

    // Apply per-item cap if it exists
    auto max_it = this->resource_reward_max.find(item);
    if (max_it != this->resource_reward_max.end()) {
      float reward_cap = max_it->second;
      old_contribution = std::min(old_contribution, reward_cap);
      new_contribution = std::min(new_contribution, reward_cap);
    }

    // Update both the current resource reward and the total reward
    float reward_delta = new_contribution - old_contribution;
    *this->reward += reward_delta;
  }

  unsigned int get_visitation_count(GridCoord r, GridCoord c) const {
    if (visitation_grid.empty() || r >= static_cast<GridCoord>(visitation_grid.size()) ||
        c >= static_cast<GridCoord>(visitation_grid[0].size())) {
      return 0;  // Return 0 for out-of-bounds positions
    }
    return visitation_grid[r][c];
  }
};

#endif  // OBJECTS_AGENT_HPP_
