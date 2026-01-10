#ifndef PACKAGES_METTAGRID_CPP_INCLUDE_METTAGRID_OBJECTS_COLLECTIVE_CHEST_HPP_
#define PACKAGES_METTAGRID_CPP_INCLUDE_METTAGRID_OBJECTS_COLLECTIVE_CHEST_HPP_

#include <algorithm>
#include <set>
#include <unordered_map>
#include <vector>

#include "config/observation_features.hpp"
#include "core/grid.hpp"
#include "core/grid_object.hpp"
#include "core/types.hpp"
#include "objects/agent.hpp"
#include "objects/alignable.hpp"
#include "objects/chest.hpp"
#include "objects/chest_config.hpp"
#include "objects/collective.hpp"
#include "objects/constants.hpp"
#include "objects/has_inventory.hpp"
#include "objects/usable.hpp"
#include "systems/observation_encoder.hpp"
#include "systems/stats_tracker.hpp"

// CollectiveChest is like Chest but uses the collective inventory instead of its own.
// It inherits from Chest and Alignable to access collective_inventory().
class CollectiveChest : public Chest, public Alignable {
private:
  // Transfer multiple resources using the COLLECTIVE inventory instead of own inventory.
  // Positive delta = deposit from agent to collective
  // Negative delta = withdraw from collective to agent
  bool transfer_resources_to_collective(Agent& agent, const std::unordered_map<InventoryItem, int>& resource_deltas) {
    Inventory* collective_inv = collective_inventory();
    if (!collective_inv) {
      return false;  // No collective assigned
    }

    bool any_transfer = false;

    for (const auto& [resource, delta] : resource_deltas) {
      if (delta > 0) {
        InventoryDelta transferred =
            HasInventory::transfer_resources(agent.inventory, *collective_inv, resource, delta, true);
        if (transferred > 0) {
          any_transfer = true;
          agent.stats.add("collective_chest." + agent.stats.resource_name(resource) + ".deposited_by_agent",
                          transferred);
        }
      } else if (delta < 0) {
        InventoryDelta transferred =
            HasInventory::transfer_resources(*collective_inv, agent.inventory, resource, -delta, true);
        if (transferred > 0) {
          any_transfer = true;
          agent.stats.add("collective_chest." + agent.stats.resource_name(resource) + ".withdrawn_by_agent",
                          transferred);
        }
      }
    }
    return any_transfer;
  }

public:
  CollectiveChest(GridCoord r, GridCoord c, const ChestConfig& cfg, StatsTracker* stats_tracker)
      : Chest(r, c, cfg, stats_tracker) {}

  virtual ~CollectiveChest() = default;

  // Override to return collective inventory instead of own inventory
  virtual Inventory* get_accessible_inventory() override {
    return collective_inventory();
  }

  // Override onUse to use collective inventory instead of own inventory
  virtual bool onUse(Agent& actor, ActionArg /*arg*/) override {
    if (!grid) {
      return false;
    }

    // Check if we have a collective assigned
    if (!getCollective()) {
      return false;
    }

    // Use vibe_transfers configuration (same as Chest, but with collective inventory)
    if (!vibe_transfers.empty()) {
      ObservationType agent_vibe = actor.vibe;
      auto vibe_it = vibe_transfers.find(agent_vibe);
      if (vibe_it == vibe_transfers.end()) {
        // Fallback to default (vibe 0) if agent's vibe not configured
        vibe_it = vibe_transfers.find(0);
      }
      if (vibe_it != vibe_transfers.end()) {
        return transfer_resources_to_collective(actor, vibe_it->second);
      }
      return false;
    }

    return false;
  }

  // Override obs_features to observe the collective inventory
  virtual std::vector<PartialObservationToken> obs_features(unsigned int observer_agent_id = UINT_MAX) const override {
    (void)observer_agent_id;  // Unused for collective_chests
    if (!this->obs_encoder) {
      throw std::runtime_error("Observation encoder not set for collective_chest");
    }

    Inventory* collective_inv = const_cast<CollectiveChest*>(this)->collective_inventory();

    std::vector<PartialObservationToken> features;

    if (this->vibe != 0) {
      features.push_back({ObservationFeature::Vibe, static_cast<ObservationType>(this->vibe)});
    }

    // Add collective inventory using multi-token encoding (if collective exists)
    if (collective_inv) {
      for (const auto& [item, amount] : collective_inv->get()) {
        if (amount > 0) {
          this->obs_encoder->append_inventory_tokens(features, item, amount);
        }
      }
    }

    // Emit tag features
    for (int tag_id : tag_ids) {
      features.push_back({ObservationFeature::Tag, static_cast<ObservationType>(tag_id)});
    }

    return features;
  }
};

#endif  // PACKAGES_METTAGRID_CPP_INCLUDE_METTAGRID_OBJECTS_COLLECTIVE_CHEST_HPP_
