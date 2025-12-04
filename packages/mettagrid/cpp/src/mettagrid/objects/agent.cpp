#include "objects/agent.hpp"

#include <algorithm>
#include <cassert>

#include "config/mettagrid_config.hpp"
#include "config/observation_features.hpp"
#include "systems/observation_encoder.hpp"

Agent::Agent(GridCoord r,
             GridCoord c,
             const AgentConfig& config,
             const std::vector<std::string>* resource_names,
             const std::unordered_map<std::string, ObservationType>* feature_ids,
             const MeleeCombatConfig* melee_combat_config)
    : GridObject(),
      HasInventory(config.inventory_config, resource_names, feature_ids),
      group(config.group_id),
      frozen(0),
      freeze_duration(config.freeze_duration),
      stat_rewards(config.stat_rewards),
      stat_reward_max(config.stat_reward_max),
      group_name(config.group_name),
      soul_bound_resources(config.soul_bound_resources),
      agent_id(0),
      stats(resource_names),
      current_stat_reward(0),
      reward(nullptr),
      prev_location(r, c),
      steps_without_motion(0),
      inventory_regen_amounts(config.inventory_regen_amounts),
      resource_names(resource_names),
      diversity_tracked_mask(resource_names != nullptr ? resource_names->size() : 0, 0),
      tracked_resource_presence(resource_names != nullptr ? resource_names->size() : 0, 0),
      tracked_resource_diversity(0),
      vibe_transfers(config.vibe_transfers) {
  if (melee_combat_config != nullptr) {
    melee_combat = *melee_combat_config;
  }

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
    this->update_inventory(item, amount);
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

bool Agent::onUse(Agent& actor, ActionArg arg) {
  // Check for melee combat
  if (actor.melee_combat.enabled && actor.melee_combat.attack_vibe_id >= 0 &&
      actor.vibe == static_cast<ObservationType>(actor.melee_combat.attack_vibe_id) &&
      actor.melee_combat.attack_item_id >= 0) {
    return handle_melee_combat(actor);
  }

  // Vibe-based resource transfer
  auto vibe_it = actor.vibe_transfers.find(actor.vibe);
  if (vibe_it == actor.vibe_transfers.end()) {
    return false;  // No transfers configured for this vibe
  }

  // Transfer each configured resource
  bool any_transfer_occurred = false;
  const auto& resource_deltas = vibe_it->second;
  for (const auto& [resource, amount] : resource_deltas) {
    if (amount > 0) {
      // Transfer from actor to receiver (this)
      InventoryQuantity actor_amount = actor.inventory.amount(resource);
      InventoryQuantity share_attempted_amount = std::min(static_cast<InventoryQuantity>(amount), actor_amount);
      if (share_attempted_amount > 0) {
        InventoryDelta successful_share_amount = this->update_inventory(resource, share_attempted_amount);
        actor.update_inventory(resource, -successful_share_amount);
        if (successful_share_amount > 0) {
          any_transfer_occurred = true;
        }
      }
    }
  }

  return any_transfer_occurred;
}

// Melee combat: attacker (in attack vibe with attack item) can freeze target and steal resources.
//
// Design notes on item consumption:
// - Attack item is consumed even when hitting an already-frozen target
// - This is intentional: it creates strategic depth by punishing wasteful attacks
// - Players should avoid attacking frozen targets to conserve ammo
// - The "wasted_on_frozen" stat tracks this for analytics
//
// Flow:
// 1. Check attacker has attack item -> fail if not (no consumption, return false)
// 2. Check target can defend -> blocked if so (return defense_consumes_item)
// 3. Consume attack item (attack committed at this point)
// 4. Freeze target and steal resources (only if not already frozen, return true)
bool Agent::handle_melee_combat(Agent& actor) {
  InventoryItem attack_item = static_cast<InventoryItem>(actor.melee_combat.attack_item_id);
  if (actor.inventory.amount(attack_item) < 1) {
    return false;  // No ammo - attack fails without consuming anything
  }

  // Defense check: target must have BOTH correct vibe AND defense item.
  // - If target has correct vibe but no item: defense fails
  // - If target has item but wrong vibe: defense fails (vibe checked first)
  // - Only when both conditions are met does defense succeed
  bool target_can_defend = false;
  if (melee_combat.defense_vibe_id >= 0 && this->vibe == static_cast<ObservationType>(melee_combat.defense_vibe_id) &&
      melee_combat.defense_item_id >= 0) {
    InventoryItem defense_item = static_cast<InventoryItem>(melee_combat.defense_item_id);
    if (this->inventory.amount(defense_item) >= 1) {
      target_can_defend = true;
      if (melee_combat.defense_consumes_item) {
        this->update_inventory(defense_item, -1);
      }
    }
  }

  if (target_can_defend) {
    log_melee_attack(actor, true);
    // Return true if defense item was consumed (state changed), false otherwise.
    // This matches the pattern in Chest/Assembler: return true iff state changed.
    return melee_combat.defense_consumes_item;
  }

  // Attack succeeds - consume ammo (even if target already frozen)
  if (actor.melee_combat.attack_consumes_item) {
    actor.update_inventory(attack_item, -1);
  }

  bool was_already_frozen = this->frozen > 0;
  this->frozen = this->freeze_duration;

  if (!was_already_frozen) {
    steal_resources(actor);
  } else {
    // Ammo was consumed but no resources stolen - track for analytics
    actor.stats.incr("action.melee." + actor.group_name + ".wasted_on_frozen");
  }

  log_melee_attack(actor, false);
  return true;
}

void Agent::steal_resources(Agent& actor) {
  // Create snapshot to avoid iterator invalidation
  std::vector<std::pair<InventoryItem, InventoryQuantity>> snapshot;
  snapshot.reserve(this->inventory.get().size());
  for (const auto& [item, amount] : this->inventory.get()) {
    snapshot.emplace_back(item, amount);
  }

  // Transfer resources (excluding soul-bound resources)
  for (const auto& [item, amount] : snapshot) {
    if (std::find(this->soul_bound_resources.begin(), this->soul_bound_resources.end(), item) !=
        this->soul_bound_resources.end()) {
      continue;
    }

    InventoryDelta stolen = actor.update_inventory(item, amount);
    this->update_inventory(item, -stolen);

    if (stolen > 0) {
      const std::string item_name = actor.stats.resource_name(item);
      const std::string stat_name =
          "action.melee." + actor.group_name + ".steals." + item_name + ".from." + this->group_name;
      actor.stats.add(stat_name, stolen);
    }
  }
}

void Agent::log_melee_attack(Agent& actor, bool blocked) {
  const std::string& actor_group = actor.group_name;
  const std::string& target_group = this->group_name;
  bool same_team = (actor_group == target_group);

  if (blocked) {
    actor.stats.incr("action.melee." + actor_group + ".blocked_by." + target_group);
  } else {
    if (same_team) {
      actor.stats.incr("action.melee." + actor_group + ".friendly_fire");
    } else {
      actor.stats.incr("action.melee." + actor_group + ".hit." + target_group);
      this->stats.incr("action.melee." + target_group + ".hit_by." + actor_group);
    }
  }
}

std::vector<PartialObservationToken> Agent::obs_features() const {
  if (!this->obs_encoder) {
    throw std::runtime_error("Observation encoder not set for agent");
  }
  const size_t num_tokens = this->inventory.get().size() + 3 + (vibe > 0 ? 1 : 0) + this->tag_ids.size();

  std::vector<PartialObservationToken> features;
  features.reserve(num_tokens);

  features.push_back({ObservationFeature::Group, static_cast<ObservationType>(group)});
  features.push_back({ObservationFeature::Frozen, static_cast<ObservationType>(frozen != 0 ? 1 : 0)});
  if (vibe != 0) features.push_back({ObservationFeature::Vibe, static_cast<ObservationType>(vibe)});

  for (const auto& [item, amount] : this->inventory.get()) {
    // inventory should only contain non-zero amounts
    assert(amount > 0);
    ObservationType item_observation_feature = this->obs_encoder->get_inventory_feature_id(item);
    features.push_back({item_observation_feature, static_cast<ObservationType>(amount)});
  }

  // Emit tag features
  for (int tag_id : tag_ids) {
    features.push_back({ObservationFeature::Tag, static_cast<ObservationType>(tag_id)});
  }

  return features;
}

void Agent::update_inventory_diversity_stats(InventoryItem item, InventoryQuantity amount) {
  const size_t index = static_cast<size_t>(item);
  if (index >= diversity_tracked_mask.size() || !diversity_tracked_mask[index]) {
    return;
  }

  const bool now_present = amount > 0;
  const bool currently_present = tracked_resource_presence[index] != 0;
  if (currently_present == now_present) {
    return;
  }

  const float prev_diversity = static_cast<float>(tracked_resource_diversity);
  tracked_resource_presence[index] = now_present ? 1 : 0;
  tracked_resource_diversity += now_present ? 1 : -1;

  const float new_diversity = static_cast<float>(tracked_resource_diversity);
  this->stats.set("inventory.diversity", new_diversity);

  for (int threshold = 2; threshold <= 5; ++threshold) {
    if (prev_diversity < static_cast<float>(threshold) && new_diversity >= static_cast<float>(threshold)) {
      this->stats.set("inventory.diversity.ge." + std::to_string(threshold), 1.0f);
    }
  }
}
