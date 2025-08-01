#ifndef OBJECTS_AGENT_HPP_
#define OBJECTS_AGENT_HPP_

#include <algorithm>
#include <cassert>
#include <string>
#include <vector>

#include "../grid_object.hpp"
#include "../stats_tracker.hpp"
#include "constants.hpp"
#include "types.hpp"

// #MettagridConfig
struct AgentConfig : public GridObjectConfig {
  AgentConfig(TypeId type_id,
              const std::string& type_name,
              unsigned char group_id,
              const std::string& group_name,
              unsigned char freeze_duration,
              float action_failure_penalty,
              const std::map<InventoryItem, InventoryQuantity>& resource_limits,
              const std::map<InventoryItem, RewardType>& resource_rewards,
              const std::map<InventoryItem, RewardType>& resource_reward_max,
              const std::map<std::string, RewardType>& stat_rewards,
              const std::map<std::string, RewardType>& stat_reward_max,
              float group_reward_pct)
      : GridObjectConfig(type_id, type_name),
        group_id(group_id),
        group_name(group_name),
        freeze_duration(freeze_duration),
        action_failure_penalty(action_failure_penalty),
        resource_limits(resource_limits),
        resource_rewards(resource_rewards),
        resource_reward_max(resource_reward_max),
        stat_rewards(stat_rewards),
        stat_reward_max(stat_reward_max),
        group_reward_pct(group_reward_pct) {}
  unsigned char group_id;
  std::string group_name;
  short freeze_duration;
  float action_failure_penalty;
  std::map<InventoryItem, InventoryQuantity> resource_limits;
  std::map<InventoryItem, RewardType> resource_rewards;
  std::map<InventoryItem, RewardType> resource_reward_max;
  std::map<std::string, RewardType> stat_rewards;
  std::map<std::string, RewardType> stat_reward_max;
  float group_reward_pct;
};

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

  // Action history tracking with ring buffer
  static constexpr size_t MAX_HISTORY_LENGTH = 1024;
  size_t history_count = 0;  // Total actions recorded (capped at MAX_HISTORY_LENGTH)

  Agent(GridCoord r, GridCoord c, const AgentConfig& config)
      : group(config.group_id),
        frozen(0),
        freeze_duration(config.freeze_duration),
        orientation(Orientation::Up),
        inventory(),  // default constructor
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
        history_count(0),
        action_history{},
        action_arg_history{},
        history_write_pos(0) {
    GridObject::init(config.type_id, config.type_name, GridLocation(r, c, GridLayer::AgentLayer));
  }

  void init(RewardType* reward_ptr) {
    this->reward = reward_ptr;
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

    // Update inventory
    if (new_amount > 0) {
      this->inventory[item] = new_amount;
    } else {
      this->inventory.erase(item);
    }

    // Update stats
    if (delta > 0) {
      this->stats.add(this->stats.inventory_item_name(item) + ".gained", static_cast<float>(delta));
    } else if (delta < 0) {
      this->stats.add(this->stats.inventory_item_name(item) + ".lost", static_cast<float>(-delta));
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

  void record_action(ActionType action, ActionArg arg) {
    action_history[history_write_pos] = action;
    action_arg_history[history_write_pos] = arg;
    // Update position and count
    history_write_pos = (history_write_pos + 1) % MAX_HISTORY_LENGTH;
    if (history_count < MAX_HISTORY_LENGTH) {
      history_count++;
    }
  }

  void copy_history_to_buffers(ActionType* action_dest, ActionArg* arg_dest) const {
    if (history_count == 0) return;

    if (history_count < MAX_HISTORY_LENGTH) {
      if (action_dest) {
        std::memcpy(action_dest, action_history.data(), history_count * sizeof(ActionType));
      }
      if (arg_dest) {
        std::memcpy(arg_dest, action_arg_history.data(), history_count * sizeof(ActionArg));
      }
    } else {
      size_t first_part = MAX_HISTORY_LENGTH - history_write_pos;
      size_t second_part = history_write_pos;

      if (action_dest) {
        std::memcpy(action_dest, action_history.data() + history_write_pos, first_part * sizeof(ActionType));
        std::memcpy(action_dest + first_part, action_history.data(), second_part * sizeof(ActionType));
      }
      if (arg_dest) {
        std::memcpy(arg_dest, action_arg_history.data() + history_write_pos, first_part * sizeof(ActionArg));
        std::memcpy(arg_dest + first_part, action_arg_history.data(), second_part * sizeof(ActionArg));
      }
    }
  }

private:
  std::array<ActionType, MAX_HISTORY_LENGTH> action_history;
  std::array<ActionArg, MAX_HISTORY_LENGTH> action_arg_history;
  size_t history_write_pos = 0;  // Current write position in ring buffer

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
};

#endif  // OBJECTS_AGENT_HPP_
