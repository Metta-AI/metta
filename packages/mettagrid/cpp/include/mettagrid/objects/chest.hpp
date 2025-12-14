#ifndef PACKAGES_METTAGRID_CPP_INCLUDE_METTAGRID_OBJECTS_CHEST_HPP_
#define PACKAGES_METTAGRID_CPP_INCLUDE_METTAGRID_OBJECTS_CHEST_HPP_

#include <algorithm>
#include <set>
#include <unordered_map>
#include <vector>

#include "config/observation_features.hpp"
#include "core/grid.hpp"
#include "core/grid_object.hpp"
#include "core/types.hpp"
#include "objects/agent.hpp"
#include "objects/chest_config.hpp"
#include "objects/constants.hpp"
#include "objects/has_inventory.hpp"
#include "objects/usable.hpp"
#include "systems/observation_encoder.hpp"
#include "systems/stats_tracker.hpp"
class Chest : public GridObject, public Usable, public HasInventory {
private:
  // a reference to the game stats tracker
  StatsTracker* stats_tracker;

  // Transfer multiple resources based on resource deltas map
  // Positive delta = deposit from agent to chest
  // Negative delta = withdraw from chest to agent
  // This method handles "as much as possible" logic for each resource:
  // - If agent tries to deposit more than they have, deposit all they have
  // - The inventory system handles max limits automatically
  bool transfer_resources(Agent& agent, const std::unordered_map<InventoryItem, int>& resource_deltas) {
    bool any_transfer = false;

    for (const auto& [resource, delta] : resource_deltas) {
      if (delta > 0) {
        InventoryDelta transferred =
            HasInventory::transfer_resources(agent.inventory, inventory, resource, delta, true);
        if (transferred > 0) {
          any_transfer = true;
          // Track per-agent chest deposits so only the actor can be rewarded
          agent.stats.add("chest." + agent.stats.resource_name(resource) + ".deposited_by_agent", transferred);
        }
      } else if (delta < 0) {
        InventoryDelta transferred =
            HasInventory::transfer_resources(inventory, agent.inventory, resource, -delta, true);
        if (transferred > 0) {
          any_transfer = true;
          // Track per-agent withdrawals (even if later disabled by config)
          agent.stats.add("chest." + agent.stats.resource_name(resource) + ".withdrawn_by_agent", transferred);
        }
      }
    }
    return any_transfer;
  }

  void on_inventory_change(InventoryItem item, InventoryDelta delta) override {
    if (delta > 0) {
      stats_tracker->add("chest." + stats_tracker->resource_name(item) + ".deposited", delta);
    } else if (delta < 0) {
      stats_tracker->add("chest." + stats_tracker->resource_name(item) + ".withdrawn", -delta);
    }
    stats_tracker->set("chest." + stats_tracker->resource_name(item) + ".amount", inventory.amount(item));
  }

public:
  // vibe -> resource -> delta
  std::unordered_map<ObservationType, std::unordered_map<InventoryItem, int>> vibe_transfers;

  // Grid access for finding agent positions
  class Grid* grid;

  // Observation encoder for inventory feature ID lookup
  const ObservationEncoder* obs_encoder = nullptr;

  Chest(GridCoord r, GridCoord c, const ChestConfig& cfg, StatsTracker* stats_tracker)
      : GridObject(),
        HasInventory(cfg.inventory_config),
        vibe_transfers(cfg.vibe_transfers),
        stats_tracker(stats_tracker),
        grid(nullptr) {
    GridObject::init(cfg.type_id, cfg.type_name, GridLocation(r, c), cfg.tag_ids, cfg.initial_vibe);
    // Set initial inventory for all configured resources (ignore limits for initial setup)
    for (const auto& [resource, amount] : cfg.initial_inventory) {
      if (amount > 0) {
        inventory.update(resource, amount, /*ignore_limits=*/true);
      }
    }
  }

  virtual ~Chest() = default;

  // Set grid access
  void set_grid(class Grid* grid_ptr) {
    this->grid = grid_ptr;
  }

  // Set observation encoder for inventory feature ID lookup
  void set_obs_encoder(const ObservationEncoder* encoder) {
    this->obs_encoder = encoder;
  }

  // Implement pure virtual method from Usable
  virtual bool onUse(Agent& actor, ActionArg /*arg*/) override {
    if (!grid) {
      return false;
    }

    // First check if vibe_transfers is configured and use it
    if (!vibe_transfers.empty()) {
      // Get the agent's current vibe
      ObservationType agent_vibe = actor.vibe;

      // Check if there's a configured resource deltas for this vibe
      auto vibe_it = vibe_transfers.find(agent_vibe);
      if (vibe_it != vibe_transfers.end()) {
        return transfer_resources(actor, vibe_it->second);
      }
      return false;  // No action configured for this vibe
    }

    return false;
  }

  virtual std::vector<PartialObservationToken> obs_features() const override {
    if (!this->obs_encoder) {
      throw std::runtime_error("Observation encoder not set for chest");
    }
    std::vector<PartialObservationToken> features;
    features.reserve(1 + this->inventory.get().size() * this->obs_encoder->get_num_inventory_tokens() +
                     this->tag_ids.size() + (this->vibe != 0 ? 1 : 0));

    if (this->vibe != 0) features.push_back({ObservationFeature::Vibe, static_cast<ObservationType>(this->vibe)});

    // Add current inventory using multi-token encoding
    for (const auto& [item, amount] : this->inventory.get()) {
      assert(amount > 0);
      this->obs_encoder->append_inventory_tokens(features, item, amount);
    }

    // Emit tag features
    for (int tag_id : tag_ids) {
      features.push_back({ObservationFeature::Tag, static_cast<ObservationType>(tag_id)});
    }

    return features;
  }
};

#endif  // PACKAGES_METTAGRID_CPP_INCLUDE_METTAGRID_OBJECTS_CHEST_HPP_
