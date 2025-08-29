#ifndef ACTIONS_TRANSFER_HPP_
#define ACTIONS_TRANSFER_HPP_

#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include <map>
#include <string>
#include <vector>

#include "action_handler.hpp"
#include "grid.hpp"
#include "grid_object.hpp"
#include "objects/agent.hpp"
#include "types.hpp"

// TransferActionConfig allows specifying exchange rates for agent-to-agent trading
struct TransferActionConfig : public ActionConfig {
  // Exchange rates: what the target agent offers in exchange for resources
  // e.g., {"ore_red": 4} -> {"battery_red": 1} means 4 ore for 1 battery
  std::map<InventoryItem, InventoryQuantity> input_resources;
  std::map<InventoryItem, InventoryQuantity> output_resources;

  // Whether the target must be a trader NPC (group_id check)
  bool trader_only;
  int trader_group_id;

  TransferActionConfig(
      const std::map<InventoryItem, InventoryQuantity>& required_resources,
      const std::map<InventoryItem, InventoryQuantity>& consumed_resources,
      const std::map<InventoryItem, InventoryQuantity>& input_resources,
      const std::map<InventoryItem, InventoryQuantity>& output_resources,
      bool trader_only = false,
      int trader_group_id = -1)
      : ActionConfig(required_resources, consumed_resources),
        input_resources(input_resources),
        output_resources(output_resources),
        trader_only(trader_only),
        trader_group_id(trader_group_id) {}
};

class Transfer : public ActionHandler {
public:
  explicit Transfer(const TransferActionConfig& cfg)
      : ActionHandler(cfg, "transfer"),
        _input_resources(cfg.input_resources),
        _output_resources(cfg.output_resources),
        _trader_only(cfg.trader_only),
        _trader_group_id(cfg.trader_group_id) {}

  unsigned char max_arg() const override {
    return 0;  // No argument needed, transfers to agent in front
  }

protected:
  std::map<InventoryItem, InventoryQuantity> _input_resources;
  std::map<InventoryItem, InventoryQuantity> _output_resources;
  bool _trader_only;
  int _trader_group_id;

  bool _handle_action(Agent* actor, ActionArg /*arg*/) override {
    // Target the square we are facing
    GridLocation target_loc = _grid->relative_location(actor->location, actor->orientation);
    target_loc.layer = GridLayer::AgentLayer;

    Agent* target_agent = static_cast<Agent*>(_grid->object_at(target_loc));
    if (!target_agent) {
      return false;
    }

    // Check if target is a valid trader (if trader_only is set)
    if (_trader_only && target_agent->group != _trader_group_id) {
      return false;
    }

    // Check if actor has required input resources
    for (const auto& [item, amount] : _input_resources) {
      auto it = actor->inventory.find(item);
      if (it == actor->inventory.end() || it->second < amount) {
        return false;
      }
    }

    // Check if target has output resources to give
    for (const auto& [item, amount] : _output_resources) {
      auto it = target_agent->inventory.find(item);
      if (it == target_agent->inventory.end() || it->second < amount) {
        // For trader NPCs, we might want infinite resources
        if (!_trader_only) {
          return false;
        }
      }
    }

    // Perform the exchange
    // Remove input resources from actor
    for (const auto& [item, amount] : _input_resources) {
      InventoryDelta delta = actor->update_inventory(item, -static_cast<InventoryDelta>(amount));
      assert(delta == -amount);
      actor->stats.add(actor->stats.inventory_item_name(item) + ".traded_away", static_cast<float>(amount));
    }

    // Add output resources to actor
    for (const auto& [item, amount] : _output_resources) {
      InventoryDelta added = actor->update_inventory(item, static_cast<InventoryDelta>(amount));
      actor->stats.add(actor->stats.inventory_item_name(item) + ".received", static_cast<float>(added));

      // Remove from target (unless it's an infinite trader)
      if (!_trader_only) {
        target_agent->update_inventory(item, -added);
      }
    }

    // Add input resources to target (unless it's an infinite trader)
    if (!_trader_only) {
      for (const auto& [item, amount] : _input_resources) {
        target_agent->update_inventory(item, static_cast<InventoryDelta>(amount));
      }
    }

    // Track successful trade (generic success is already incremented by ActionHandler)
    if (_trader_only) {
      actor->stats.incr("action.transfer.with_trader");
    } else {
      actor->stats.incr("action.transfer.with_agent");
    }

    return true;
  }
};

namespace py = pybind11;

inline void bind_transfer_action_config(py::module& m) {
  py::class_<TransferActionConfig, ActionConfig, std::shared_ptr<TransferActionConfig>>(m, "TransferActionConfig")
      .def(py::init<const std::map<InventoryItem, InventoryQuantity>&,
                    const std::map<InventoryItem, InventoryQuantity>&,
                    const std::map<InventoryItem, InventoryQuantity>&,
                    const std::map<InventoryItem, InventoryQuantity>&,
                    bool,
                    int>(),
           py::arg("required_resources") = std::map<InventoryItem, InventoryQuantity>(),
           py::arg("consumed_resources") = std::map<InventoryItem, InventoryQuantity>(),
           py::arg("input_resources") = std::map<InventoryItem, InventoryQuantity>(),
           py::arg("output_resources") = std::map<InventoryItem, InventoryQuantity>(),
           py::arg("trader_only") = false,
           py::arg("trader_group_id") = -1)
      .def_readwrite("input_resources", &TransferActionConfig::input_resources)
      .def_readwrite("output_resources", &TransferActionConfig::output_resources)
      .def_readwrite("trader_only", &TransferActionConfig::trader_only)
      .def_readwrite("trader_group_id", &TransferActionConfig::trader_group_id);
}

#endif  // ACTIONS_TRANSFER_HPP_
