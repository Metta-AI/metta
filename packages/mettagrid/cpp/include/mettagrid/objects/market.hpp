#ifndef PACKAGES_METTAGRID_CPP_INCLUDE_METTAGRID_OBJECTS_MARKET_HPP_
#define PACKAGES_METTAGRID_CPP_INCLUDE_METTAGRID_OBJECTS_MARKET_HPP_

#include <algorithm>
#include <cassert>
#include <cmath>
#include <set>
#include <unordered_map>
#include <vector>

#include "actions/orientation.hpp"
#include "config/observation_features.hpp"
#include "core/grid.hpp"
#include "core/grid_object.hpp"
#include "core/types.hpp"
#include "objects/agent.hpp"
#include "objects/constants.hpp"
#include "objects/has_inventory.hpp"
#include "objects/market_config.hpp"
#include "objects/usable.hpp"
#include "systems/observation_encoder.hpp"
#include "systems/stats_tracker.hpp"

class Market : public GridObject, public Usable, public HasInventory {
private:
  StatsTracker* stats_tracker;

  // Terminal configurations by direction
  std::unordered_map<int, MarketTerminalConfig> terminals;

  // Currency resource ID (e.g., hearts)
  InventoryItem currency_resource_id;

  // Maps vibe ID -> resource ID for trading
  std::unordered_map<ObservationType, InventoryItem> vibe_to_resource;

  // Vibe names for debug output
  std::vector<std::string> vibe_names;

  void on_inventory_change(InventoryItem item, InventoryDelta delta) override {
    if (delta > 0) {
      stats_tracker->add("market." + stats_tracker->resource_name(item) + ".bought", delta);
    } else if (delta < 0) {
      stats_tracker->add("market." + stats_tracker->resource_name(item) + ".sold", -delta);
    }
    stats_tracker->set("market." + stats_tracker->resource_name(item) + ".amount", inventory.amount(item));
  }

  // Calculate price for a resource: 100 / sqrt(inventory)
  // Returns price in hearts (rounded to nearest integer)
  // Only call this when a trade is possible (item in stock)
  int calculate_price(InventoryItem resource) const {
    InventoryQuantity current_amount = inventory.amount(resource);
    assert(current_amount > 0 && "calculate_price called but item not in stock");
    return static_cast<int>(std::round(100.0 / std::sqrt(static_cast<double>(current_amount))));
  }

  // Get vibe name for debug output
  std::string get_vibe_name(ObservationType vibe_id) const {
    if (vibe_id < vibe_names.size()) {
      return vibe_names[vibe_id];
    }
    return "vibe_" + std::to_string(vibe_id);
  }

  // Get resource name for debug output
  std::string get_resource_name(InventoryItem resource_id) const {
    return stats_tracker->resource_name(resource_id);
  }

public:
  class Grid* grid;
  const ObservationEncoder* obs_encoder = nullptr;

  Market(GridCoord r, GridCoord c, const MarketConfig& cfg, StatsTracker* stats_tracker)
      : GridObject(),
        HasInventory(cfg.inventory_config),
        stats_tracker(stats_tracker),
        terminals(cfg.terminals),
        currency_resource_id(cfg.currency_resource_id),
        vibe_to_resource(cfg.vibe_to_resource),
        vibe_names(cfg.vibe_names),
        grid(nullptr) {
    const DemolishConfig* demolish = cfg.demolish.has_value() ? &cfg.demolish.value() : nullptr;
    GridObject::init(cfg.type_id, cfg.type_name, GridLocation(r, c), cfg.tag_ids, cfg.initial_vibe, demolish, cfg.aoe);

    // Set initial inventory for all configured resources
    for (const auto& [resource, amount] : cfg.initial_inventory) {
      if (amount > 0) {
        inventory.update(resource, amount);
      }
    }
  }

  virtual ~Market() = default;

  void set_grid(class Grid* grid_ptr) {
    this->grid = grid_ptr;
  }

  void set_obs_encoder(const ObservationEncoder* encoder) {
    this->obs_encoder = encoder;
  }

private:
  virtual bool onUse(Agent& actor, ActionArg arg) override {
    if (!grid) {
      return false;
    }

    // Get the move direction from arg
    Orientation move_direction = static_cast<Orientation>(arg);

    // The terminal is based on the direction the agent came FROM
    Orientation source_direction = opposite_direction(move_direction);

    std::cout << "[Market] Agent " << actor.agent_id << " entered from direction " << static_cast<int>(source_direction)
              << " (moved " << static_cast<int>(move_direction) << ")" << std::endl;

    // Find terminal config for this direction
    auto terminal_it = terminals.find(static_cast<int>(source_direction));
    if (terminal_it == terminals.end()) {
      std::cout << "[Market] No terminal on direction " << static_cast<int>(source_direction) << std::endl;
      return false;  // No terminal on this side
    }

    const MarketTerminalConfig& terminal = terminal_it->second;
    std::cout << "[Market] Terminal config: " << (terminal.sell ? "SELL" : "BUY") << ", amount=" << terminal.amount
              << std::endl;

    // Get the resource the agent is vibing
    ObservationType agent_vibe = actor.vibe;
    std::cout << "[Market] Agent vibe: " << get_vibe_name(agent_vibe) << std::endl;

    // Look up the resource ID for this vibe
    auto vibe_it = vibe_to_resource.find(agent_vibe);
    if (vibe_it == vibe_to_resource.end()) {
      std::cout << "[Market] ABORT: Vibe '" << get_vibe_name(agent_vibe) << "' has no tradeable resource" << std::endl;
      return false;
    }

    InventoryItem resource_item = vibe_it->second;
    std::cout << "[Market] Vibe '" << get_vibe_name(agent_vibe) << "' maps to resource '"
              << get_resource_name(resource_item) << "'" << std::endl;

    // Hearts can't be bought or sold
    if (resource_item == currency_resource_id) {
      std::cout << "[Market] ABORT: Can't trade " << get_resource_name(currency_resource_id) << " (currency)"
                << std::endl;
      return false;
    }

    bool any_trade = false;

    // Process one resource at a time, up to terminal.amount
    for (int i = 0; i < terminal.amount; ++i) {
      std::cout << "[Market] Trade iteration " << i + 1 << "/" << terminal.amount << std::endl;
      if (terminal.sell) {
        // Agent is SELLING to the market
        // Agent gives resource, receives hearts
        std::string resource_name = get_resource_name(resource_item);
        std::string currency_name = get_resource_name(currency_resource_id);

        // === VALIDATION PHASE - check everything before any changes ===
        InventoryQuantity agent_has = actor.inventory.amount(resource_item);
        std::cout << "[Market SELL] Agent " << actor.agent_id << " wants to sell " << resource_name << ", has "
                  << static_cast<int>(agent_has) << std::endl;
        if (agent_has == 0) {
          std::cout << "[Market SELL] ABORT: Agent has no " << resource_name << " to sell" << std::endl;
          break;
        }

        // Check market has space for the resource
        InventoryQuantity market_free_space = inventory.free_space(resource_item);
        std::cout << "[Market SELL] Market has " << static_cast<int>(market_free_space) << " free space for "
                  << resource_name << std::endl;
        if (market_free_space == 0) {
          std::cout << "[Market SELL] ABORT: Market inventory full for " << resource_name << std::endl;
          break;
        }

        // Calculate price based on what market will have AFTER receiving the item
        InventoryQuantity market_will_have = inventory.amount(resource_item) + 1;
        int price = static_cast<int>(std::round(100.0 / std::sqrt(static_cast<double>(market_will_have))));
        std::cout << "[Market SELL] Price: 100/sqrt(" << static_cast<int>(market_will_have) << ") = " << price << " "
                  << currency_name << std::endl;

        // Check agent has space for hearts
        InventoryQuantity agent_heart_space = actor.inventory.free_space(currency_resource_id);
        std::cout << "[Market SELL] Agent has " << static_cast<int>(agent_heart_space) << " free space for "
                  << currency_name << std::endl;
        if (agent_heart_space < static_cast<InventoryQuantity>(price)) {
          std::cout << "[Market SELL] ABORT: Agent can't hold " << price << " " << currency_name << " (only space for "
                    << static_cast<int>(agent_heart_space) << ")" << std::endl;
          break;
        }

        // === EXECUTION PHASE - all checks passed, do the trade ===
        actor.inventory.update(resource_item, -1);
        inventory.update(resource_item, 1);
        actor.inventory.update(currency_resource_id, price);

        std::cout << "[Market SELL] Trade complete! Agent: " << resource_name << "="
                  << static_cast<int>(actor.inventory.amount(resource_item)) << ", " << currency_name << "="
                  << static_cast<int>(actor.inventory.amount(currency_resource_id)) << std::endl;

        stats_tracker->add("market.trades.sell", 1);
        stats_tracker->add("market.hearts.paid", price);
        any_trade = true;
      } else {
        // Agent is BUYING from the market
        // Agent gives hearts, receives resource
        std::string resource_name = get_resource_name(resource_item);
        std::string currency_name = get_resource_name(currency_resource_id);

        // === VALIDATION PHASE - check everything before any changes ===
        InventoryQuantity market_has = inventory.amount(resource_item);
        std::cout << "[Market BUY] Agent " << actor.agent_id << " wants " << resource_name << ", market has "
                  << static_cast<int>(market_has) << std::endl;
        if (market_has == 0) {
          std::cout << "[Market BUY] ABORT: Market has no " << resource_name << std::endl;
          break;
        }

        // Calculate price when we know item is in stock
        int price = calculate_price(resource_item);
        std::cout << "[Market BUY] Price: 100/sqrt(" << static_cast<int>(market_has) << ") = " << price << " "
                  << currency_name << std::endl;

        InventoryQuantity agent_hearts = actor.inventory.amount(currency_resource_id);
        std::cout << "[Market BUY] Agent has " << static_cast<int>(agent_hearts) << " " << currency_name << ", needs "
                  << price << std::endl;
        if (agent_hearts < static_cast<InventoryQuantity>(price)) {
          std::cout << "[Market BUY] ABORT: Agent can't afford " << price << " " << currency_name << std::endl;
          break;
        }

        // Check if agent can receive the resource
        InventoryQuantity agent_free_space = actor.inventory.free_space(resource_item);
        std::cout << "[Market BUY] Agent has " << static_cast<int>(agent_free_space) << " free space for "
                  << resource_name << std::endl;
        if (agent_free_space == 0) {
          std::cout << "[Market BUY] ABORT: Agent inventory full for " << resource_name << std::endl;
          break;
        }

        // === EXECUTION PHASE - all checks passed, do the trade ===
        actor.inventory.update(currency_resource_id, -price);
        inventory.update(resource_item, -1);
        actor.inventory.update(resource_item, 1);

        std::cout << "[Market BUY] Trade complete! Agent: " << resource_name << "="
                  << static_cast<int>(actor.inventory.amount(resource_item)) << ", " << currency_name << "="
                  << static_cast<int>(actor.inventory.amount(currency_resource_id)) << std::endl;

        stats_tracker->add("market.trades.buy", 1);
        stats_tracker->add("market.hearts.received", price);
        any_trade = true;
      }
    }

    return any_trade;
  }

  virtual std::vector<PartialObservationToken> obs_features(unsigned int observer_agent_id = UINT_MAX) const override {
    (void)observer_agent_id;
    if (!this->obs_encoder) {
      throw std::runtime_error("Observation encoder not set for market");
    }
    std::vector<PartialObservationToken> features;
    features.reserve(1 + this->inventory.get().size() * 2 + this->tag_ids.size() + (this->vibe != 0 ? 1 : 0));

    if (this->vibe != 0) features.push_back({ObservationFeature::Vibe, static_cast<ObservationType>(this->vibe)});

    // Add current inventory (inv:resource)
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

#endif  // PACKAGES_METTAGRID_CPP_INCLUDE_METTAGRID_OBJECTS_MARKET_HPP_
