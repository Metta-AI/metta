#ifndef PACKAGES_METTAGRID_CPP_INCLUDE_METTAGRID_OBJECTS_COMMONS_CHEST_HPP_
#define PACKAGES_METTAGRID_CPP_INCLUDE_METTAGRID_OBJECTS_COMMONS_CHEST_HPP_

#include <algorithm>
#include <set>
#include <unordered_map>
#include <vector>

#include "config/observation_features.hpp"
#include "core/grid.hpp"
#include "core/grid_object.hpp"
#include "core/types.hpp"
#include "objects/agent.hpp"
#include "objects/chest.hpp"
#include "objects/chest_config.hpp"
#include "objects/commons.hpp"
#include "objects/constants.hpp"
#include "objects/has_inventory.hpp"
#include "objects/usable.hpp"
#include "systems/observation_encoder.hpp"
#include "systems/stats_tracker.hpp"

// CommonsChest is like Chest but uses the commons inventory instead of its own.
// It inherits from Chest but overrides methods to use commons_inventory().
class CommonsChest : public Chest {
private:
  // Transfer multiple resources using the COMMONS inventory instead of own inventory.
  // Positive delta = deposit from agent to commons
  // Negative delta = withdraw from commons to agent
  bool transfer_resources_to_commons(Agent& agent, const std::unordered_map<InventoryItem, int>& resource_deltas) {
    Inventory* commons_inv = commons_inventory();
    if (!commons_inv) {
      return false;  // No commons assigned
    }

    bool any_transfer = false;

    for (const auto& [resource, delta] : resource_deltas) {
      if (delta > 0) {
        InventoryDelta transferred =
            HasInventory::transfer_resources(agent.inventory, *commons_inv, resource, delta, true);
        if (transferred > 0) {
          any_transfer = true;
          agent.stats.add("commons_chest." + agent.stats.resource_name(resource) + ".deposited_by_agent", transferred);
        }
      } else if (delta < 0) {
        InventoryDelta transferred =
            HasInventory::transfer_resources(*commons_inv, agent.inventory, resource, -delta, true);
        if (transferred > 0) {
          any_transfer = true;
          agent.stats.add("commons_chest." + agent.stats.resource_name(resource) + ".withdrawn_by_agent", transferred);
        }
      }
    }
    return any_transfer;
  }

public:
  CommonsChest(GridCoord r, GridCoord c, const ChestConfig& cfg, StatsTracker* stats_tracker)
      : Chest(r, c, cfg, stats_tracker) {}

  virtual ~CommonsChest() = default;

  // Override to return commons inventory instead of own inventory
  virtual Inventory* get_accessible_inventory() override {
    return commons_inventory();
  }

  // Override onUse to use commons inventory instead of own inventory
  virtual bool onUse(Agent& actor, ActionArg /*arg*/) override {
    if (!grid) {
      return false;
    }

    // Check if we have a commons assigned
    if (!getCommons()) {
      return false;
    }

    // Use vibe_transfers configuration (same as Chest, but with commons inventory)
    if (!vibe_transfers.empty()) {
      ObservationType agent_vibe = actor.vibe;
      auto vibe_it = vibe_transfers.find(agent_vibe);
      if (vibe_it == vibe_transfers.end()) {
        // Fallback to default (vibe 0) if agent's vibe not configured
        vibe_it = vibe_transfers.find(0);
      }
      if (vibe_it != vibe_transfers.end()) {
        return transfer_resources_to_commons(actor, vibe_it->second);
      }
      return false;
    }

    return false;
  }

  // Override obs_features to observe the commons inventory
  virtual std::vector<PartialObservationToken> obs_features() const override {
    if (!this->obs_encoder) {
      throw std::runtime_error("Observation encoder not set for commons_chest");
    }

    Inventory* commons_inv = const_cast<CommonsChest*>(this)->commons_inventory();

    std::vector<PartialObservationToken> features;

    if (this->vibe != 0) {
      features.push_back({ObservationFeature::Vibe, static_cast<ObservationType>(this->vibe)});
    }

    // Add commons inventory using multi-token encoding (if commons exists)
    if (commons_inv) {
      for (const auto& [item, amount] : commons_inv->get()) {
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

#endif  // PACKAGES_METTAGRID_CPP_INCLUDE_METTAGRID_OBJECTS_COMMONS_CHEST_HPP_
