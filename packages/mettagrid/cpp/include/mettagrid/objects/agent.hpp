#ifndef PACKAGES_METTAGRID_CPP_INCLUDE_METTAGRID_OBJECTS_AGENT_HPP_
#define PACKAGES_METTAGRID_CPP_INCLUDE_METTAGRID_OBJECTS_AGENT_HPP_

#include <algorithm>
#include <array>
#include <cassert>
#include <random>
#include <string>
#include <vector>

#include "core/types.hpp"
#include "objects/agent_config.hpp"
#include "objects/alignable.hpp"
#include "objects/constants.hpp"
#include "objects/has_inventory.hpp"
#include "objects/usable.hpp"
#include "systems/stats_tracker.hpp"

class ObservationEncoder;

class Agent : public GridObject, public HasInventory, public Usable, public Alignable {
public:
  ObservationType group;
  short frozen;
  short freeze_duration;
  // inventory is a map of item to amount.
  // keys should be deleted when the amount is 0, to keep iteration faster.
  // however, this should not be relied on for correctness.
  std::unordered_map<std::string, RewardType> stat_rewards;
  std::unordered_map<std::string, RewardType> stat_reward_max;
  std::string group_name;
  // Despite being a GridObjectId, this is different from the `id` property.
  // This is the index into MettaGrid._agents (std::vector<Agent*>)
  GridObjectId agent_id;
  StatsTracker stats;
  RewardType current_stat_reward;
  RewardType* reward;
  GridLocation prev_location;
  unsigned int steps_without_motion;
  // Vibe-dependent inventory regeneration: vibe_id -> resource_id -> amount (can be negative for decay)
  // Vibe ID 0 ("default") is used as fallback when agent's current vibe is not found
  std::unordered_map<ObservationType, std::unordered_map<InventoryItem, InventoryDelta>> inventory_regen_amounts;
  // Damage configuration
  DamageConfig damage_config;

  Agent(GridCoord r,
        GridCoord c,
        const AgentConfig& config,
        const std::vector<std::string>* resource_names,
        const std::unordered_map<std::string, ObservationType>* feature_ids = nullptr);

  void init(RewardType* reward_ptr);

  void populate_initial_inventory(const std::unordered_map<InventoryItem, InventoryQuantity>& initial_inventory);

  void set_inventory(const std::unordered_map<InventoryItem, InventoryQuantity>& inventory);

  void on_inventory_change(InventoryItem item, InventoryDelta delta) override;

  void compute_stat_rewards(StatsTracker* game_stats_tracker = nullptr);

  // Check and apply damage if all threshold stats are reached
  // Returns true if damage was applied
  bool check_and_apply_damage(std::mt19937& rng);

  // Implementation of Usable interface
  bool onUse(Agent& actor, ActionArg arg) override;

  std::vector<PartialObservationToken> obs_features() const override;

  // Set observation encoder for inventory feature ID lookup
  void set_obs_encoder(const ObservationEncoder* encoder) {
    this->obs_encoder = encoder;
  }

private:
  const ObservationEncoder* obs_encoder = nullptr;
  const std::vector<std::string>* resource_names = nullptr;
  void update_inventory_diversity_stats(InventoryItem item, InventoryQuantity amount);
  std::vector<char> diversity_tracked_mask;
  std::vector<char> tracked_resource_presence;
  std::size_t tracked_resource_diversity{0};
};

#endif  // PACKAGES_METTAGRID_CPP_INCLUDE_METTAGRID_OBJECTS_AGENT_HPP_
