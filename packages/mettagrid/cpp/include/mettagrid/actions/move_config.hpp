// move_config.hpp
#ifndef PACKAGES_METTAGRID_CPP_INCLUDE_METTAGRID_ACTIONS_MOVE_CONFIG_HPP_
#define PACKAGES_METTAGRID_CPP_INCLUDE_METTAGRID_ACTIONS_MOVE_CONFIG_HPP_

#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include <string>
#include <vector>

#include "actions/action_handler.hpp"
#include "core/types.hpp"

struct MoveActionConfig : public ActionConfig {
  std::vector<std::string> allowed_directions;

  MoveActionConfig(const std::vector<std::string>& allowed_directions = {"north", "south", "west", "east"},
                   const std::unordered_map<InventoryItem, InventoryQuantity>& required_resources = {},
                   const std::unordered_map<InventoryItem, InventoryQuantity>& consumed_resources = {})
      : ActionConfig(required_resources, consumed_resources), allowed_directions(allowed_directions) {}
};

namespace py = pybind11;

inline void bind_move_action_config(py::module& m) {
  py::class_<MoveActionConfig, ActionConfig, std::shared_ptr<MoveActionConfig>>(m, "MoveActionConfig")
      .def(py::init<const std::vector<std::string>&,
                    const std::unordered_map<InventoryItem, InventoryQuantity>&,
                    const std::unordered_map<InventoryItem, InventoryQuantity>&>(),
           py::arg("allowed_directions") = std::vector<std::string>{"north", "south", "west", "east"},
           py::arg("required_resources") = std::unordered_map<InventoryItem, InventoryQuantity>(),
           py::arg("consumed_resources") = std::unordered_map<InventoryItem, InventoryQuantity>())
      .def_readwrite("allowed_directions", &MoveActionConfig::allowed_directions);
}

#endif  // PACKAGES_METTAGRID_CPP_INCLUDE_METTAGRID_ACTIONS_MOVE_CONFIG_HPP_
