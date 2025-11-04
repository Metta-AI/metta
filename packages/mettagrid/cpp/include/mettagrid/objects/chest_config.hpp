#ifndef PACKAGES_METTAGRID_CPP_INCLUDE_METTAGRID_OBJECTS_CHEST_CONFIG_HPP_
#define PACKAGES_METTAGRID_CPP_INCLUDE_METTAGRID_OBJECTS_CHEST_CONFIG_HPP_

#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include <set>
#include <unordered_map>

#include "core/grid_object.hpp"
#include "core/types.hpp"
#include "objects/inventory_config.hpp"

struct ChestConfig : public GridObjectConfig {
  ChestConfig(TypeId type_id, const std::string& type_name, ObservationType initial_vibe = 0)
      : GridObjectConfig(type_id, type_name, initial_vibe),
        vibe_transfers({}),
        initial_inventory({}),
        inventory_config() {}

  // Maps vibe to resource deltas (positive = deposit, negative = withdraw)
  std::unordered_map<ObservationType, std::unordered_map<InventoryItem, int>> vibe_transfers;

  // Initial inventory for each resource type
  std::unordered_map<InventoryItem, int> initial_inventory;

  // Inventory configuration with limits
  InventoryConfig inventory_config;
};

namespace py = pybind11;

inline void bind_chest_config(py::module& m) {
  py::class_<ChestConfig, GridObjectConfig, std::shared_ptr<ChestConfig>>(m, "ChestConfig")
      .def(py::init<TypeId, const std::string&, ObservationType>(),
           py::arg("type_id"),
           py::arg("type_name"),
           py::arg("initial_vibe") = 0)
      .def_readwrite("type_id", &ChestConfig::type_id)
      .def_readwrite("type_name", &ChestConfig::type_name)
      .def_readwrite("tag_ids", &ChestConfig::tag_ids)
      .def_readwrite("vibe_transfers", &ChestConfig::vibe_transfers)
      .def_readwrite("initial_inventory", &ChestConfig::initial_inventory)
      .def_readwrite("inventory_config", &ChestConfig::inventory_config)
      .def_readwrite("initial_vibe", &ChestConfig::initial_vibe);
}

#endif  // PACKAGES_METTAGRID_CPP_INCLUDE_METTAGRID_OBJECTS_CHEST_CONFIG_HPP_
