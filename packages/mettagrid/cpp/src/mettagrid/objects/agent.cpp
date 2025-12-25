#include "objects/agent.hpp"

#include <algorithm>
#include <cassert>

#include "config/observation_features.hpp"
#include "systems/observation_encoder.hpp"

// For std::shuffle
#include <random>

Agent::Agent(GridCoord r,
             GridCoord c,
             const AgentConfig& config,
             const std::vector<std::string>* resource_names,
             const std::unordered_map<std::string, ObservationType>* feature_ids)
    : GridObject(),
      HasInventory(config.inventory_config, resource_names, feature_ids),
      group(config.group_id),
      frozen(0),
      freeze_duration(config.freeze_duration),
      stat_rewards(config.stat_rewards),
      stat_reward_max(config.stat_reward_max),
      group_name(config.group_name),
      agent_id(0),
      stats(resource_names),
      current_stat_reward(0),
      reward(nullptr),
      prev_location(r, c),
      steps_without_motion(0),
      inventory_regen_amounts(config.inventory_regen_amounts),
      damage_config(config.damage_config),
      resource_names(resource_names),
      diversity_tracked_mask(resource_names != nullptr ? resource_names->size() : 0, 0),
      tracked_resource_presence(resource_names != nullptr ? resource_names->size() : 0, 0),
      tracked_resource_diversity(0) {
  for (InventoryItem item : config.diversity_tracked_resources) {
    const size_t index = static_cast<size_t>(item);
    if (index < diversity_tracked_mask.size()) {
      diversity_tracked_mask[index] = 1;
    }
  }
  populate_initial_inventory(config.initial_inventory);
  GridObject::init(config.type_id, config.type_name, GridLocation(r, c), config.tag_ids, config.initial_vibe);
}

void Agent::init(RewardType* reward_ptr) {
  this->reward = reward_ptr;
}

void Agent::populate_initial_inventory(const std::unordered_map<InventoryItem, InventoryQuantity>& initial_inventory) {
  for (const auto& [item, amount] : initial_inventory) {
    this->inventory.update(item, amount, /*ignore_limits=*/true);
  }
}

void Agent::set_inventory(const std::unordered_map<InventoryItem, InventoryQuantity>& inventory) {
  // First, remove items that are not present in the provided inventory map
  // Make a copy of current item keys to avoid iterator invalidation
  std::vector<InventoryItem> existing_items;
  for (const auto& [existing_item, existing_amount] : this->inventory.get()) {
    existing_items.push_back(existing_item);
  }

  for (const auto& existing_item : existing_items) {
    const InventoryQuantity current_amount = this->inventory.amount(existing_item);
    this->inventory.update(existing_item, -static_cast<InventoryDelta>(current_amount));
    this->stats.set(this->stats.resource_name(existing_item) + ".amount", 0);
  }

  // Then, set provided items to their specified amounts
  for (const auto& [item, amount] : inventory) {
    this->inventory.update(item, amount - this->inventory.amount(item));
  }
}

void Agent::on_inventory_change(InventoryItem item, InventoryDelta delta) {
  const InventoryQuantity amount = this->inventory.amount(item);
  if (delta != 0) {
    if (delta > 0) {
      this->stats.add(this->stats.resource_name(item) + ".gained", delta);
    } else if (delta < 0) {
      this->stats.add(this->stats.resource_name(item) + ".lost", -delta);
    }
    this->stats.set(this->stats.resource_name(item) + ".amount", amount);
  }
  update_inventory_diversity_stats(item, amount);
}

void Agent::update_inventory_diversity_stats(InventoryItem item, InventoryQuantity amount) {
  const size_t index = static_cast<size_t>(item);
  if (index >= this->diversity_tracked_mask.size() || this->diversity_tracked_mask[index] == 0) {
    return;
  }

  const bool had = this->tracked_resource_presence[index] != 0;
  const bool has = amount > 0;

  if (had != has) {
    this->tracked_resource_presence[index] = has ? 1 : 0;
    this->tracked_resource_diversity += has ? 1 : static_cast<std::size_t>(-1);
    this->stats.set("inventory.diversity", static_cast<float>(this->tracked_resource_diversity));
  }
}

void Agent::compute_stat_rewards(StatsTracker* game_stats_tracker) {
  if (this->stat_rewards.empty()) {
    return;
  }

  float new_stat_reward = 0;

  for (const auto& [stat_name, reward_per_unit] : this->stat_rewards) {
    float stat_value = this->stats.get(stat_name);
    if (game_stats_tracker) {
      stat_value += game_stats_tracker->get(stat_name);
    }
    float stats_reward = stat_value * reward_per_unit;
    if (this->stat_reward_max.count(stat_name) > 0) {
      stats_reward = std::min(stats_reward, this->stat_reward_max.at(stat_name));
    }

    new_stat_reward += stats_reward;
  }

  // Update the agent's reward with the difference
  float reward_delta = new_stat_reward - this->current_stat_reward;
  if (reward_delta != 0.0f) {
    *this->reward += reward_delta;
    this->current_stat_reward = new_stat_reward;
  }
}

bool Agent::check_and_apply_damage(std::mt19937& rng) {
  if (!damage_config.enabled()) {
    return false;
  }

  // Check if all threshold inventory items are at or above their threshold values
  for (const auto& [item, threshold_value] : damage_config.threshold) {
    InventoryQuantity amount = this->inventory.amount(item);
    if (amount < static_cast<InventoryQuantity>(threshold_value)) {
      return false;  // Not all thresholds met
    }
  }

  // Subtract threshold values from inventory first
  for (const auto& [item, threshold_value] : damage_config.threshold) {
    this->inventory.update(item, -static_cast<InventoryDelta>(threshold_value));
  }

  // Find which resources from the damage map the agent has above their minimum
  // and build weights based on quantity available for removal (after threshold subtraction)
  std::vector<InventoryItem> available_resources;
  std::vector<int> weights;
  for (const auto& [item, minimum] : damage_config.resources) {
    InventoryQuantity amount = this->inventory.amount(item);
    int removable = static_cast<int>(amount) - minimum;
    if (removable > 0) {
      available_resources.push_back(item);
      weights.push_back(removable);
    }
  }

  // If resources available, pick one weighted by quantity above minimum
  if (!available_resources.empty()) {
    std::discrete_distribution<size_t> dist(weights.begin(), weights.end());
    size_t selected_idx = dist(rng);
    InventoryItem item_to_remove = available_resources[selected_idx];
    this->inventory.update(item_to_remove, -1);
    this->stats.incr("damage.items_lost");
    this->stats.incr("damaged." + this->stats.resource_name(item_to_remove));
  }

  this->stats.incr("damage.triggered");
  return true;
}

bool Agent::onUse(Agent& actor, ActionArg arg) {
  // Agent-to-agent transfers are now handled by the Transfer action handler.
  // This method returns false to indicate no default use action.
  (void)actor;
  (void)arg;
  return false;
}

std::vector<PartialObservationToken> Agent::obs_features() const {
  if (!this->obs_encoder) {
    throw std::runtime_error("Observation encoder not set for agent");
  }
  const size_t num_tokens =
      this->inventory.get().size() * this->obs_encoder->get_num_inventory_tokens() + this->tag_ids.size() + 5;

  std::vector<PartialObservationToken> features;
  features.reserve(num_tokens);

  features.push_back({ObservationFeature::Group, static_cast<ObservationType>(group)});
  features.push_back({ObservationFeature::Frozen, static_cast<ObservationType>(frozen != 0 ? 1 : 0)});
  if (vibe != 0) features.push_back({ObservationFeature::Vibe, static_cast<ObservationType>(vibe)});

  for (const auto& [item, amount] : this->inventory.get()) {
    // inventory should only contain non-zero amounts
    assert(amount > 0);
    this->obs_encoder->append_inventory_tokens(features, item, amount);
  }

  // Emit tag features
  for (int tag_id : tag_ids) {
    features.push_back({ObservationFeature::Tag, static_cast<ObservationType>(tag_id)});
  }

  return features;
}
