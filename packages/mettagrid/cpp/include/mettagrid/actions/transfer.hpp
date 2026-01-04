#ifndef PACKAGES_METTAGRID_CPP_INCLUDE_METTAGRID_ACTIONS_TRANSFER_HPP_
#define PACKAGES_METTAGRID_CPP_INCLUDE_METTAGRID_ACTIONS_TRANSFER_HPP_

#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include <string>
#include <unordered_map>
#include <vector>

#include "actions/action_handler.hpp"
#include "core/grid_object.hpp"
#include "core/types.hpp"
#include "objects/agent.hpp"
#include "objects/has_inventory.hpp"

namespace py = pybind11;

// Holds the resource deltas for both actor and target in a vibe transfer
struct VibeTransferEffect {
  // Resource deltas applied to the target (e.g., {energy: 10} means target gains 10 energy)
  std::unordered_map<InventoryItem, int> target_deltas;
  // Resource deltas applied to the actor (e.g., {energy: -10, heart: -1} means actor loses resources)
  std::unordered_map<InventoryItem, int> actor_deltas;

  VibeTransferEffect() = default;
  VibeTransferEffect(const std::unordered_map<InventoryItem, int>& target,
                     const std::unordered_map<InventoryItem, int>& actor)
      : target_deltas(target), actor_deltas(actor) {}
};

struct TransferActionConfig : public ActionConfig {
  // Maps vibe ID to transfer effects (separate actor/target deltas)
  std::unordered_map<ObservationType, VibeTransferEffect> vibe_transfers;
  bool enabled;

  TransferActionConfig(const std::unordered_map<InventoryItem, InventoryQuantity>& required_resources = {},
                       const std::unordered_map<ObservationType, VibeTransferEffect>& vibe_transfers = {},
                       bool enabled = true)
      : ActionConfig(required_resources, {}), vibe_transfers(vibe_transfers), enabled(enabled) {}
};

class Transfer : public ActionHandler {
public:
  explicit Transfer(const TransferActionConfig& cfg,
                    [[maybe_unused]] const GameConfig* game_config,
                    const std::string& action_name = "transfer")
      : ActionHandler(cfg, action_name), _vibe_transfers(cfg.vibe_transfers), _enabled(cfg.enabled) {
    priority = 0;  // Lower priority than attack
    // Derive vibes from vibe_transfers keys
    for (const auto& [vibe_id, _] : _vibe_transfers) {
      _vibes.push_back(vibe_id);
    }
  }

  std::vector<Action> create_actions() override {
    // Transfer doesn't create standalone actions - it's triggered by move
    return {};
  }

  // Get vibes that trigger this action on move (derived from vibe_transfers)
  const std::vector<ObservationType>& get_vibes() const {
    return _vibes;
  }

  // Check if the actor's vibe has a transfer configured
  bool has_transfer_for_vibe(ObservationType vibe) const {
    return _vibe_transfers.find(vibe) != _vibe_transfers.end();
  }

  // Expose to Move class - transfer decides if target is valid
  bool try_transfer(Agent& actor, GridObject* target_object) {
    // Check if actor has the required resources
    for (const auto& [item, amount] : _required_resources) {
      if (actor.inventory.amount(item) < amount) {
        return false;
      }
    }
    if (!target_object) return false;

    Agent* target = dynamic_cast<Agent*>(target_object);
    if (!target) return false;             // Can only transfer with agents
    if (target->frozen > 0) return false;  // Can't transfer with frozen agents, allow swap

    auto vibe_it = _vibe_transfers.find(actor.vibe);
    if (vibe_it == _vibe_transfers.end()) {
      return false;  // No transfer configured for this vibe
    }

    const VibeTransferEffect& effect = vibe_it->second;
    const std::string& actor_group = actor.group_name;
    const std::string& target_group = target->group_name;

    // 1. Check if actor has resources to give (negative deltas)
    // Cast inventory amounts to int to avoid truncation when delta exceeds uint16_t range
    for (const auto& [resource, delta] : effect.actor_deltas) {
      if (delta < 0 && static_cast<int>(actor.inventory.amount(resource)) < -delta) {
        return false;  // Actor doesn't have enough resources to give
      }
    }

    // 2. Check if target has resources to give (negative deltas)
    for (const auto& [resource, delta] : effect.target_deltas) {
      if (delta < 0 && static_cast<int>(target->inventory.amount(resource)) < -delta) {
        return false;  // Target doesn't have enough resources to give
      }
    }

    // 3. Check if actor has capacity for receiving resources (positive deltas)
    for (const auto& [resource, delta] : effect.actor_deltas) {
      if (delta > 0) {
        int free = static_cast<int>(actor.inventory.free_space(resource));
        if (delta > free) {
          return false;  // Actor doesn't have capacity to receive
        }
      }
    }

    // 4. Check if target has capacity for receiving resources (positive deltas)
    for (const auto& [resource, delta] : effect.target_deltas) {
      if (delta > 0) {
        int free = static_cast<int>(target->inventory.free_space(resource));
        if (delta > free) {
          return false;  // Target doesn't have capacity to receive
        }
      }
    }

    // 5. Update actor and target resources
    for (const auto& [resource, delta] : effect.actor_deltas) {
      if (delta != 0) {
        InventoryDelta actual = actor.inventory.update(resource, delta);
        if (actual != 0) {
          _log_transfer(actor, resource, actual, "self");
        }
      }
    }

    for (const auto& [resource, delta] : effect.target_deltas) {
      if (delta != 0) {
        InventoryDelta actual = target->inventory.update(resource, delta);
        if (actual != 0) {
          _log_transfer(actor, resource, actual, "to." + target_group);
        }
      }
    }

    actor.stats.incr(_action_prefix(actor_group) + "count");
    return true;
  }

protected:
  std::unordered_map<ObservationType, VibeTransferEffect> _vibe_transfers;
  bool _enabled;
  std::vector<ObservationType> _vibes;

  bool _handle_action(Agent& actor, ActionArg arg) override {
    // Transfer is not called directly as an action
    (void)actor;
    (void)arg;
    return false;
  }

private:
  std::string _action_prefix(const std::string& group) const {
    return "action." + _action_name + "." + group + ".";
  }

  void _log_transfer(Agent& actor, InventoryItem item, InventoryDelta amount, const std::string& direction) const {
    const std::string& actor_group = actor.group_name;
    const std::string item_name = actor.stats.resource_name(item);

    // Track the actual amount (positive = gained, negative = lost)
    actor.stats.add(_action_prefix(actor_group) + item_name + "." + direction, amount);
  }
};

inline void bind_vibe_transfer_effect(py::module& m) {
  py::class_<VibeTransferEffect>(m, "VibeTransferEffect")
      .def(py::init<>())
      .def(py::init<const std::unordered_map<InventoryItem, int>&, const std::unordered_map<InventoryItem, int>&>(),
           py::arg("target_deltas") = std::unordered_map<InventoryItem, int>(),
           py::arg("actor_deltas") = std::unordered_map<InventoryItem, int>())
      .def_readwrite("target_deltas", &VibeTransferEffect::target_deltas)
      .def_readwrite("actor_deltas", &VibeTransferEffect::actor_deltas);
}

inline void bind_transfer_action_config(py::module& m) {
  py::class_<TransferActionConfig, ActionConfig, std::shared_ptr<TransferActionConfig>>(m, "TransferActionConfig")
      .def(py::init<const std::unordered_map<InventoryItem, InventoryQuantity>&,
                    const std::unordered_map<ObservationType, VibeTransferEffect>&,
                    bool>(),
           py::arg("required_resources") = std::unordered_map<InventoryItem, InventoryQuantity>(),
           py::arg("vibe_transfers") = std::unordered_map<ObservationType, VibeTransferEffect>(),
           py::arg("enabled") = true)
      .def_readwrite("vibe_transfers", &TransferActionConfig::vibe_transfers)
      .def_readwrite("enabled", &TransferActionConfig::enabled);
}

#endif  // PACKAGES_METTAGRID_CPP_INCLUDE_METTAGRID_ACTIONS_TRANSFER_HPP_
