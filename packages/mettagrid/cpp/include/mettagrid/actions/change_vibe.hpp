#ifndef PACKAGES_METTAGRID_CPP_INCLUDE_METTAGRID_ACTIONS_CHANGE_VIBE_HPP_
#define PACKAGES_METTAGRID_CPP_INCLUDE_METTAGRID_ACTIONS_CHANGE_VIBE_HPP_

#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include <string>

#include "actions/action_handler.hpp"
#include "core/types.hpp"
#include "objects/agent.hpp"

struct ChangeVibeActionConfig : public ActionConfig {
  const ObservationType number_of_vibes;

  ChangeVibeActionConfig(const std::unordered_map<InventoryItem, InventoryQuantity>& required_resources,
                         const std::unordered_map<InventoryItem, InventoryProbability>& consumed_resources,
                         const ObservationType number_of_vibes)
      : ActionConfig(required_resources, consumed_resources), number_of_vibes(number_of_vibes) {}
};

class ChangeVibe : public ActionHandler {
public:
  explicit ChangeVibe(const ChangeVibeActionConfig& cfg)
      : ActionHandler(cfg, "change_vibe"), _number_of_vibes(cfg.number_of_vibes) {}

  unsigned char max_arg() const override {
    // Return number_of_vibes - 1 since args are 0-indexed
    return _number_of_vibes > 0 ? _number_of_vibes - 1 : 0;
  }

protected:
  const ObservationType _number_of_vibes;

  bool _handle_action(Agent& actor, ActionArg arg) override {
    actor.set_vibe(static_cast<ObservationType>(arg));  // ActionArg is int32 for puffer compatibility
    return true;
  }
};

namespace py = pybind11;

inline void bind_change_vibe_action_config(py::module& m) {
  py::class_<ChangeVibeActionConfig, ActionConfig, std::shared_ptr<ChangeVibeActionConfig>>(m, "ChangeVibeActionConfig")
      .def(py::init<const std::unordered_map<InventoryItem, InventoryQuantity>&,
                    const std::unordered_map<InventoryItem, InventoryProbability>&,
                    const int>(),
           py::arg("required_resources") = std::unordered_map<InventoryItem, InventoryQuantity>(),
           py::arg("consumed_resources") = std::unordered_map<InventoryItem, InventoryProbability>(),
           py::arg("number_of_vibes"))
      .def_readonly("number_of_vibes", &ChangeVibeActionConfig::number_of_vibes);
}
#endif  // PACKAGES_METTAGRID_CPP_INCLUDE_METTAGRID_ACTIONS_CHANGE_VIBE_HPP_
