#ifndef PACKAGES_METTAGRID_CPP_INCLUDE_METTAGRID_ACTIONS_ALIGN_HPP_
#define PACKAGES_METTAGRID_CPP_INCLUDE_METTAGRID_ACTIONS_ALIGN_HPP_

#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include <string>
#include <unordered_map>
#include <vector>

#include "actions/action_handler.hpp"
#include "core/grid_object.hpp"
#include "core/types.hpp"
#include "objects/agent.hpp"
#include "objects/alignable.hpp"
#include "objects/commons.hpp"

namespace py = pybind11;

struct AlignActionConfig : public ActionConfig {
  // Vibe that triggers align on move
  ObservationType vibe;
  // Cost to the actor for aligning
  std::unordered_map<InventoryItem, int> cost;
  // Cost deducted from the actor's commons
  std::unordered_map<InventoryItem, int> commons_cost;
  // If true, sets target's commons to none instead of actor's commons (scramble mode)
  bool set_to_none;

  AlignActionConfig(const std::unordered_map<InventoryItem, InventoryQuantity>& required_resources = {},
                    ObservationType vibe = 0,
                    const std::unordered_map<InventoryItem, int>& cost = {},
                    const std::unordered_map<InventoryItem, int>& commons_cost = {},
                    bool set_to_none = false)
      : ActionConfig(required_resources, {}),
        vibe(vibe),
        cost(cost),
        commons_cost(commons_cost),
        set_to_none(set_to_none) {}
};

class Align : public ActionHandler {
public:
  explicit Align(const AlignActionConfig& cfg,
                 [[maybe_unused]] const GameConfig* game_config,
                 const std::string& action_name = "align")
      : ActionHandler(cfg, action_name),
        _vibe(cfg.vibe),
        _cost(cfg.cost),
        _commons_cost(cfg.commons_cost),
        _set_to_none(cfg.set_to_none) {
    priority = 0;  // Same priority as transfer
  }

  std::vector<Action> create_actions() override {
    // Align doesn't create standalone actions - it's triggered by move
    return {};
  }

  // Get the vibe that triggers this action on move
  ObservationType get_vibe() const {
    return _vibe;
  }

  // Expose to Move class - align decides if target is valid
  bool try_align(Agent& actor, GridObject* target_object) {
    if (!target_object) return false;

    // Check if actor's vibe matches the align vibe
    if (actor.vibe != _vibe) return false;

    // Check if actor has the required resources
    for (const auto& [item, amount] : _required_resources) {
      if (actor.inventory.amount(item) < amount) {
        return false;
      }
    }

    // Check if actor can pay the cost
    for (const auto& [resource, amount] : _cost) {
      if (amount > 0 && static_cast<int>(actor.inventory.amount(resource)) < amount) {
        return false;
      }
    }

    // Check if actor's commons can pay the commons cost
    Commons* actor_commons = actor.getCommons();
    if (!_commons_cost.empty()) {
      if (!actor_commons) return false;
      Inventory* commons_inv = actor.commons_inventory();
      if (!commons_inv) return false;
      for (const auto& [resource, amount] : _commons_cost) {
        if (amount > 0 && static_cast<int>(commons_inv->amount(resource)) < amount) {
          return false;
        }
      }
    }

    const std::string& actor_group = actor.group_name;

    // Try to align the target
    Alignable* alignable_target = dynamic_cast<Alignable*>(target_object);
    if (!alignable_target) return false;

    Commons* target_commons = alignable_target->getCommons();
    Commons* new_commons = _set_to_none ? nullptr : actor_commons;

    // Check validity based on mode
    if (_set_to_none) {
      // For scramble: target must have a commons to clear
      if (target_commons == nullptr) {
        return false;
      }
    } else {
      // For align: actor must have commons, and target must be unaligned (no commons)
      // To align an enemy-aligned target, must first scramble it to neutral
      if (actor_commons == nullptr || target_commons != nullptr) {
        return false;
      }
    }

    // Pay the costs
    for (const auto& [resource, amount] : _cost) {
      if (amount != 0) {
        actor.inventory.update(resource, -amount);
      }
    }
    if (actor_commons && !_commons_cost.empty()) {
      Inventory* commons_inv = actor.commons_inventory();
      if (commons_inv) {
        for (const auto& [resource, amount] : _commons_cost) {
          if (amount != 0) {
            commons_inv->update(resource, -amount);
          }
        }
      }
    }

    // Perform the alignment
    alignable_target->setCommons(new_commons);
    actor.stats.incr(_action_prefix(actor_group) + (_set_to_none ? "scrambled" : "aligned"));
    actor.stats.incr(_action_prefix(actor_group) + "count");

    return true;
  }

protected:
  ObservationType _vibe;
  std::unordered_map<InventoryItem, int> _cost;
  std::unordered_map<InventoryItem, int> _commons_cost;
  bool _set_to_none;

  bool _handle_action(Agent& actor, ActionArg arg) override {
    // Align is not called directly as an action
    (void)actor;
    (void)arg;
    return false;
  }

private:
  std::string _action_prefix(const std::string& group) const {
    return "action." + _action_name + "." + group + ".";
  }
};

inline void bind_align_action_config(py::module& m) {
  py::class_<AlignActionConfig, ActionConfig, std::shared_ptr<AlignActionConfig>>(m, "AlignActionConfig")
      .def(py::init<const std::unordered_map<InventoryItem, InventoryQuantity>&,
                    ObservationType,
                    const std::unordered_map<InventoryItem, int>&,
                    const std::unordered_map<InventoryItem, int>&,
                    bool>(),
           py::arg("required_resources") = std::unordered_map<InventoryItem, InventoryQuantity>(),
           py::arg("vibe") = 0,
           py::arg("cost") = std::unordered_map<InventoryItem, int>(),
           py::arg("commons_cost") = std::unordered_map<InventoryItem, int>(),
           py::arg("set_to_none") = false)
      .def_readwrite("vibe", &AlignActionConfig::vibe)
      .def_readwrite("cost", &AlignActionConfig::cost)
      .def_readwrite("commons_cost", &AlignActionConfig::commons_cost)
      .def_readwrite("set_to_none", &AlignActionConfig::set_to_none);
}

#endif  // PACKAGES_METTAGRID_CPP_INCLUDE_METTAGRID_ACTIONS_ALIGN_HPP_
