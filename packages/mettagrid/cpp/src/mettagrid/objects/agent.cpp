#include "objects/agent.hpp"

#include <algorithm>
#include <cassert>

#include "config/observation_features.hpp"

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
      action_failure_penalty(config.action_failure_penalty),
      group_name(config.group_name),
      soul_bound_resources(config.soul_bound_resources),
      shareable_resources(config.shareable_resources),
      agent_id(0),
      stats(resource_names),
      current_stat_reward(0),
      reward(nullptr),
      prev_location(r, c, GridLayer::AgentLayer),
      prev_action_name(""),
      steps_without_motion(0),
      inventory_regen_amounts(config.inventory_regen_amounts),
      diversity_tracked_mask_(resource_names != nullptr ? resource_names->size() : 0, 0),
      tracked_resource_presence_(resource_names != nullptr ? resource_names->size() : 0, 0),
      tracked_resource_diversity_(0) {
  for (InventoryItem item : config.diversity_tracked_resources) {
    const size_t index = static_cast<size_t>(item);
    if (index < diversity_tracked_mask_.size()) {
      diversity_tracked_mask_[index] = 1;
    }
  }

  populate_initial_inventory(config.initial_inventory);
  GridObject::init(
      config.type_id, config.type_name, GridLocation(r, c, GridLayer::AgentLayer), config.tag_ids, config.initial_vibe);
}

void Agent::init(RewardType* reward_ptr) {
  this->reward = reward_ptr;
}

void Agent::populate_initial_inventory(const std::unordered_map<InventoryItem, InventoryQuantity>& initial_inventory) {
  for (const auto& [item, amount] : initial_inventory) {
    this->update_inventory(item, amount);
  }
}

void Agent::init_visitation_grid(GridCoord height, GridCoord width) {
  visitation_grid.resize(height, std::vector<unsigned int>(width, 0));
  visitation_counts_enabled = true;
}

void Agent::reset_visitation_counts() {
  for (auto& row : visitation_grid) {
    std::fill(row.begin(), row.end(), 0);
  }
}

void Agent::increment_visitation_count(GridCoord r, GridCoord c) {
  if (!visitation_counts_enabled) return;

  if (r < static_cast<GridCoord>(visitation_grid.size()) && c < static_cast<GridCoord>(visitation_grid[0].size())) {
    visitation_grid[r][c]++;
  }
}

std::array<unsigned int, 5> Agent::get_visitation_counts() const {
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
    update_inventory_diversity_stats(existing_item, 0);
  }

  // Then, set provided items to their specified amounts
  for (const auto& [item, amount] : inventory) {
    // Go through update_inventory to handle limits, deal with rewards, etc.
    this->update_inventory(item, amount - this->inventory.amount(item));
  }
}

InventoryDelta Agent::update_inventory(InventoryItem item, InventoryDelta attempted_delta) {
  const InventoryDelta delta = this->inventory.update(item, attempted_delta);
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

  return delta;
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

bool Agent::swappable() const {
  return this->frozen;
}

bool Agent::onUse(Agent& actor, ActionArg arg) {
  // Share half of shareable resources from actor to this agent
  for (InventoryItem resource : actor.shareable_resources) {
    InventoryQuantity actor_amount = actor.inventory.amount(resource);
    // Calculate half (rounded down)
    InventoryQuantity share_attempted_amount = actor_amount / 2;
    if (share_attempted_amount > 0) {
      // The actor is trying to give us resources. We need to make sure we can take them.
      InventoryDelta successful_share_amount = this->update_inventory(resource, share_attempted_amount);
      actor.update_inventory(resource, -successful_share_amount);
    }
  }

  return true;
}

std::vector<PartialObservationToken> Agent::obs_features() const {
  const size_t num_tokens = this->inventory.get().size() + 4 + (vibe > 0 ? 1 : 0) + this->tag_ids.size();

  std::vector<PartialObservationToken> features;
  features.reserve(num_tokens);

  features.push_back({ObservationFeature::TypeId, static_cast<ObservationType>(type_id)});
  features.push_back({ObservationFeature::Group, static_cast<ObservationType>(group)});
  features.push_back({ObservationFeature::Frozen, static_cast<ObservationType>(frozen != 0 ? 1 : 0)});
  if (vibe != 0) features.push_back({ObservationFeature::Vibe, static_cast<ObservationType>(vibe)});

  for (const auto& [item, amount] : this->inventory.get()) {
    // inventory should only contain non-zero amounts
    assert(amount > 0);
    auto item_observation_feature = this->inventory.get_feature_id(item);
    features.push_back({item_observation_feature, static_cast<ObservationType>(amount)});
  }

  // Emit tag features
  for (int tag_id : tag_ids) {
    features.push_back({ObservationFeature::Tag, static_cast<ObservationType>(tag_id)});
  }

  return features;
}

unsigned int Agent::get_visitation_count(GridCoord r, GridCoord c) const {
  if (visitation_grid.empty() || r >= static_cast<GridCoord>(visitation_grid.size()) ||
      c >= static_cast<GridCoord>(visitation_grid[0].size())) {
    return 0;  // Return 0 for out-of-bounds positions
  }
  return visitation_grid[r][c];
}

void Agent::update_inventory_diversity_stats(InventoryItem item, InventoryQuantity amount) {
  const size_t index = static_cast<size_t>(item);
  if (index >= diversity_tracked_mask_.size() || !diversity_tracked_mask_[index]) {
    return;
  }

  const bool now_present = amount > 0;
  const bool currently_present = tracked_resource_presence_[index] != 0;
  if (currently_present == now_present) {
    return;
  }

  const float prev_diversity = static_cast<float>(tracked_resource_diversity_);
  tracked_resource_presence_[index] = now_present ? 1 : 0;
  tracked_resource_diversity_ += now_present ? 1 : -1;

  const float new_diversity = static_cast<float>(tracked_resource_diversity_);
  this->stats.set("inventory.diversity", new_diversity);

  for (int threshold = 2; threshold <= 5; ++threshold) {
    if (prev_diversity < static_cast<float>(threshold) && new_diversity >= static_cast<float>(threshold)) {
      this->stats.set("inventory.diversity.ge." + std::to_string(threshold), 1.0f);
    }
  }
}
