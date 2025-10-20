#ifndef PACKAGES_METTAGRID_CPP_INCLUDE_METTAGRID_OBJECTS_CHEST_CONFIG_HPP_
#define PACKAGES_METTAGRID_CPP_INCLUDE_METTAGRID_OBJECTS_CHEST_CONFIG_HPP_

#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include <set>
#include <unordered_map>

#include "core/grid_object.hpp"
#include "core/types.hpp"

struct ChestConfig : public GridObjectConfig {
  ChestConfig(TypeId type_id,
              const std::string& type_name,
              InventoryItem resource_type,
              const std::unordered_map<int, int>& position_deltas,
              int initial_inventory = 0,
              int max_inventory = 255,
              const std::vector<int>& tag_ids = {})
      : GridObjectConfig(type_id, type_name, tag_ids),
        resource_type(resource_type),
        position_deltas(position_deltas),
        initial_inventory(initial_inventory),
        max_inventory(max_inventory) {}

  InventoryItem resource_type;
  std::unordered_map<int, int>
      position_deltas;    // position_index -> delta (positive = deposit amount, negative = withdraw amount)
  int initial_inventory;  // Initial amount of resource_type in the chest
  int max_inventory;      // Maximum inventory (-1 = unlimited)
};

namespace py = pybind11;

inline void bind_chest_config(py::module& m) {
  py::class_<ChestConfig, GridObjectConfig, std::shared_ptr<ChestConfig>>(m, "ChestConfig")
      .def(py::init<TypeId,
                    const std::string&,
                    InventoryItem,
                    const std::unordered_map<int, int>&,
                    int,
                    int,
                    const std::vector<int>&>(),
           py::arg("type_id"),
           py::arg("type_name"),
           py::arg("resource_type"),
           py::arg("position_deltas"),
           py::arg("initial_inventory") = 0,
           py::arg("max_inventory") = 255,
           py::arg("tag_ids") = std::vector<int>())
      .def_readwrite("type_id", &ChestConfig::type_id)
      .def_readwrite("type_name", &ChestConfig::type_name)
      .def_readwrite("resource_type", &ChestConfig::resource_type)
      .def_readwrite("position_deltas", &ChestConfig::position_deltas)
      .def_readwrite("initial_inventory", &ChestConfig::initial_inventory)
      .def_readwrite("max_inventory", &ChestConfig::max_inventory)
      .def_readwrite("tag_ids", &ChestConfig::tag_ids);
}

#endif  // PACKAGES_METTAGRID_CPP_INCLUDE_METTAGRID_OBJECTS_CHEST_CONFIG_HPP_
