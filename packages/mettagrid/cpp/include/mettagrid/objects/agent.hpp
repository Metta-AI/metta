#ifndef PACKAGES_METTAGRID_CPP_INCLUDE_METTAGRID_OBJECTS_AGENT_HPP_
#define PACKAGES_METTAGRID_CPP_INCLUDE_METTAGRID_OBJECTS_AGENT_HPP_

#include <algorithm>
#include <array>
#include <cassert>
#include <string>
#include <vector>

#include "actions/orientation.hpp"
#include "core/types.hpp"
#include "objects/constants.hpp"
#include "objects/has_inventory.hpp"
#include "objects/usable.hpp"
#include "supervisors/agent_supervisor.hpp"
#include "systems/stats_tracker.hpp"

class AgentConfig;
class Agent : public GridObject, public HasInventory, public Usable {
public:
  ObservationType group;
  short frozen;
  short freeze_duration;
  Orientation orientation;
  // inventory is a map of item to amount.
  // keys should be deleted when the amount is 0, to keep iteration faster.
  // however, this should not be relied on for correctness.
  std::unordered_map<std::string, RewardType> stat_rewards;
  std::unordered_map<std::string, RewardType> stat_reward_max;
  float action_failure_penalty;
  std::string group_name;
  // We expect only a small number (single-digit) of soul-bound resources.
  std::vector<InventoryItem> soul_bound_resources;
  // Resources that this agent will try to share when it uses another agent.
  std::vector<InventoryItem> shareable_resources;
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
  // Inventory regeneration amounts (per-agent)
  std::unordered_map<InventoryItem, InventoryQuantity> inventory_regen_amounts;
  // Agent supervisor (optional)
  std::unique_ptr<AgentSupervisor> supervisor;

  Agent(GridCoord r, GridCoord c, const AgentConfig& config, const std::vector<std::string>* resource_names);

  void init(RewardType* reward_ptr);

  void populate_initial_inventory(const std::unordered_map<InventoryItem, InventoryQuantity>& initial_inventory);

  void init_visitation_grid(GridCoord height, GridCoord width);

  void reset_visitation_counts();

  void increment_visitation_count(GridCoord r, GridCoord c);

  std::array<unsigned int, 5> get_visitation_counts() const;

  void set_inventory(const std::unordered_map<InventoryItem, InventoryQuantity>& inventory);

  InventoryDelta update_inventory(InventoryItem item, InventoryDelta attempted_delta);

  void compute_stat_rewards(StatsTracker* game_stats_tracker = nullptr);

  bool swappable() const override;

  // Implementation of Usable interface
  bool onUse(Agent& actor, ActionArg arg) override;

  std::vector<PartialObservationToken> obs_features() const override;

private:
  unsigned int get_visitation_count(GridCoord r, GridCoord c) const;
  void update_inventory_diversity_stats(InventoryItem item, InventoryQuantity amount);
  std::vector<char> diversity_tracked_mask_;
  std::vector<char> tracked_resource_presence_;
  std::size_t tracked_resource_diversity_{0};
};

#endif  // PACKAGES_METTAGRID_CPP_INCLUDE_METTAGRID_OBJECTS_AGENT_HPP_
