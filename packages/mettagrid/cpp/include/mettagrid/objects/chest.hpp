#ifndef PACKAGES_METTAGRID_CPP_INCLUDE_METTAGRID_OBJECTS_CHEST_HPP_
#define PACKAGES_METTAGRID_CPP_INCLUDE_METTAGRID_OBJECTS_CHEST_HPP_

#include <set>
#include <unordered_map>
#include <vector>

#include "core/event.hpp"
#include "core/grid.hpp"
#include "core/grid_object.hpp"
#include "core/types.hpp"
#include "objects/agent.hpp"
#include "objects/chest_config.hpp"
#include "objects/constants.hpp"
#include "objects/has_inventory.hpp"
#include "objects/usable.hpp"
#include "systems/stats_tracker.hpp"

class Chest : public GridObject, public Usable, public HasInventory {
private:
  // a reference to the game stats tracker
  StatsTracker* stats_tracker;
  // Get the relative position index of the agent from the chest
  // Returns bit index: NW=0, N=1, NE=2, W=3, E=4, SW=5, S=6, SE=7
  int get_agent_relative_position_index(const Agent& agent) const {
    if (!grid) return -1;

    GridCoord chest_r = location.r;
    GridCoord chest_c = location.c;
    GridCoord agent_r = agent.location.r;
    GridCoord agent_c = agent.location.c;

    int delta_r = agent_r - chest_r;
    int delta_c = agent_c - chest_c;

    // Map delta coordinates to position indices (matching assembler pattern)
    if (delta_r == -1 && delta_c == -1) return 0;  // NW
    if (delta_r == -1 && delta_c == 0) return 1;   // N
    if (delta_r == -1 && delta_c == 1) return 2;   // NE
    if (delta_r == 0 && delta_c == -1) return 3;   // W
    if (delta_r == 0 && delta_c == 1) return 4;    // E
    if (delta_r == 1 && delta_c == -1) return 5;   // SW
    if (delta_r == 1 && delta_c == 0) return 6;    // S
    if (delta_r == 1 && delta_c == 1) return 7;    // SE

    return -1;  // Agent is not adjacent
  }

  // Transfer resources based on position delta
  // Positive delta = deposit from agent to chest
  // Negative delta = withdraw from chest to agent
  bool transfer_resource(Agent& agent, int delta) {
    if (delta > 0) {
      // Deposit: agent -> chest
      int agent_amount = static_cast<int>(agent.inventory.amount(resource_type));
      int transfer_amount = std::min(delta, agent_amount);
      if (transfer_amount == 0) {
        return false;
      }

      // Check if chest has space (unless max_inventory is unlimited)
      if (max_inventory >= 0) {
        int current_amount = static_cast<int>(inventory.amount(resource_type));
        int available_space = max_inventory - current_amount;
        transfer_amount = std::min(transfer_amount, available_space);

        if (transfer_amount == 0) {
          // Chest is full, destroy the resource from agent
          agent.update_inventory(resource_type, -delta);
          stats_tracker->add("chest." + stats_tracker->resource_name(resource_type) + ".destroyed", delta);
          return true;
        }
      }

      // Transfer from agent to chest
      InventoryDelta deposited = update_inventory(resource_type, transfer_amount);
      agent.update_inventory(resource_type, -transfer_amount);
      stats_tracker->add("chest." + stats_tracker->resource_name(resource_type) + ".deposited", transfer_amount);
      stats_tracker->set("chest." + stats_tracker->resource_name(resource_type) + ".amount",
                         inventory.amount(resource_type));

      // If we couldn't transfer the full delta due to max_inventory, destroy the rest
      int destroyed = delta - transfer_amount;
      if (destroyed > 0) {
        agent.update_inventory(resource_type, -destroyed);
        stats_tracker->add("chest." + stats_tracker->resource_name(resource_type) + ".destroyed", destroyed);
      }
      return true;
    } else if (delta < 0) {
      // Withdraw: chest -> agent
      int chest_amount = static_cast<int>(inventory.amount(resource_type));
      int withdraw_amount = std::min(-delta, chest_amount);
      if (withdraw_amount == 0) {
        return false;
      }

      // Transfer from chest to agent
      InventoryDelta withdrawn = agent.update_inventory(resource_type, withdraw_amount);
      if (withdrawn > 0) {
        update_inventory(resource_type, -withdrawn);
        stats_tracker->add("chest." + stats_tracker->resource_name(resource_type) + ".withdrawn", withdrawn);
        stats_tracker->set("chest." + stats_tracker->resource_name(resource_type) + ".amount",
                           inventory.amount(resource_type));
        return true;
      }
      return false;
    }
    return false;  // delta == 0, no-op
  }

public:
  // Configuration
  InventoryItem resource_type;
  std::unordered_map<int, int> position_deltas;  // position_index -> delta
  int max_inventory;                             // Maximum inventory (-1 = unlimited)

  // Grid access for finding agent positions
  class Grid* grid;

  Chest(GridCoord r, GridCoord c, const ChestConfig& cfg, StatsTracker* stats_tracker)
      : GridObject(),
        HasInventory(InventoryConfig()),  // Chests have nothing to configure in their inventory. Yet.
        resource_type(cfg.resource_type),
        position_deltas(cfg.position_deltas),
        max_inventory(cfg.max_inventory),
        stats_tracker(stats_tracker),
        grid(nullptr) {
    GridObject::init(cfg.type_id, cfg.type_name, GridLocation(r, c, GridLayer::ObjectLayer), cfg.tag_ids);
    // Set initial inventory
    if (cfg.initial_inventory > 0) {
      update_inventory(cfg.resource_type, cfg.initial_inventory);
    }
  }

  virtual ~Chest() = default;

  // Set grid access
  void set_grid(class Grid* grid_ptr) {
    this->grid = grid_ptr;
  }

  // Implement pure virtual method from Usable
  virtual bool onUse(Agent& actor, ActionArg /*arg*/) override {
    if (!grid) {
      return false;
    }

    int agent_position_index = get_agent_relative_position_index(actor);
    if (agent_position_index == -1) {
      return false;
    }

    // Check if there's a configured delta for this position
    auto it = position_deltas.find(agent_position_index);
    if (it == position_deltas.end()) {
      return false;  // No action configured for this position
    }

    int delta = it->second;
    return transfer_resource(actor, delta);
  }

  virtual std::vector<PartialObservationToken> obs_features() const override {
    std::vector<PartialObservationToken> features;
    features.reserve(2 + this->inventory.get().size() + this->tag_ids.size());

    features.push_back({ObservationFeature::TypeId, static_cast<ObservationType>(this->type_id)});

    // Add current inventory (inv:resource)
    for (const auto& [item, amount] : this->inventory.get()) {
      if (amount > 0) {
        features.push_back(
            {static_cast<ObservationType>(item + InventoryFeatureOffset), static_cast<ObservationType>(amount)});
      }
    }

    // Emit tag features
    for (int tag_id : tag_ids) {
      features.push_back({ObservationFeature::Tag, static_cast<ObservationType>(tag_id)});
    }

    return features;
  }
};

#endif  // PACKAGES_METTAGRID_CPP_INCLUDE_METTAGRID_OBJECTS_CHEST_HPP_
