#ifndef PACKAGES_METTAGRID_CPP_INCLUDE_METTAGRID_OBJECTS_CHEST_HPP_
#define PACKAGES_METTAGRID_CPP_INCLUDE_METTAGRID_OBJECTS_CHEST_HPP_

#include <map>
#include <set>
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

// Forward declaration
class Agent;

class Chest : public GridObject, public Usable, public HasInventory {
private:
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

  // Check if the given position index is a deposit position
  bool is_deposit_position(int position_index) const {
    return deposit_positions.find(position_index) != deposit_positions.end();
  }

  // Check if the given position index is a withdrawal position
  bool is_withdrawal_position(int position_index) const {
    return withdrawal_positions.find(position_index) != withdrawal_positions.end();
  }

  // Deposit a resource from agent to chest
  bool deposit_resource(Agent& agent) {
    // Check if agent has the required resource
    auto it = agent.inventory.find(resource_type);
    if (it == agent.inventory.end() || it->second == 0) {
      return false;
    }

    InventoryDelta deposited = update_inventory(resource_type, 1);
    if (deposited == 1) {
      agent.update_inventory(resource_type, -1);
      return true;
    }
    // Chest couldn't accept the resource, give it back to agent
    return false;
  }

  // Withdraw a resource from chest to agent
  bool withdraw_resource(Agent& agent) {
    // Check if chest has the required resource
    auto it = inventory.find(resource_type);
    if (it == inventory.end() || it->second == 0) {
      return false;
    }

    InventoryDelta withdrawn = agent.update_inventory(resource_type, 1);
    if (withdrawn == 1) {
      update_inventory(resource_type, -1);
      return true;
    }
    // Agent couldn't accept the resource, give it back to chest
    return false;
  }

public:
  // Configuration
  InventoryItem resource_type;
  std::set<int> deposit_positions;
  std::set<int> withdrawal_positions;

  // Grid access for finding agent positions
  class Grid* grid;

  Chest(GridCoord r, GridCoord c, const ChestConfig& cfg)
      : resource_type(cfg.resource_type),
        deposit_positions(cfg.deposit_positions),
        withdrawal_positions(cfg.withdrawal_positions),
        grid(nullptr) {
    GridObject::init(cfg.type_id, cfg.type_name, GridLocation(r, c, GridLayer::ObjectLayer), cfg.tag_ids);
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

    // Check if agent is in a valid position for deposit or withdrawal
    if (is_deposit_position(agent_position_index)) {
      return deposit_resource(actor);
    } else if (is_withdrawal_position(agent_position_index)) {
      return withdraw_resource(actor);
    } else {
      return false;
    }
  }

  virtual std::vector<PartialObservationToken> obs_features() const override {
    std::vector<PartialObservationToken> features;
    features.reserve(2 + this->inventory.size() + this->tag_ids.size());

    features.push_back({ObservationFeature::TypeId, static_cast<ObservationType>(this->type_id)});
    features.push_back({ObservationFeature::Color, static_cast<ObservationType>(this->resource_type)});

    // Add current inventory (inv:resource)
    for (const auto& [item, amount] : this->inventory) {
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
