#ifndef PACKAGES_METTAGRID_CPP_INCLUDE_METTAGRID_OBJECTS_CHEST_CONFIG_HPP_
#define PACKAGES_METTAGRID_CPP_INCLUDE_METTAGRID_OBJECTS_CHEST_CONFIG_HPP_

#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include <set>
#include <unordered_map>

#include "core/grid_object.hpp"
#include "core/types.hpp"

struct ChestConfig : public GridObjectConfig {
  ChestConfig(TypeId type_id, const std::string& type_name)
      : GridObjectConfig(type_id, type_name),
        resource_type(0),
        position_deltas({}),
        initial_inventory(0),
        max_inventory(255) {}

  InventoryItem resource_type;
  std::unordered_map<int, int>
      position_deltas;    // position_index -> delta (positive = deposit amount, negative = withdraw amount)
  int initial_inventory;  // Initial amount of resource_type in the chest
  int max_inventory;      // Maximum inventory (-1 = unlimited)
};

namespace py = pybind11;

inline void bind_chest_config(py::module& m) {
  py::class_<ChestConfig, GridObjectConfig, std::shared_ptr<ChestConfig>>(m, "ChestConfig")
      .def(py::init<TypeId, const std::string&>(), py::arg("type_id"), py::arg("type_name"))
      .def_readwrite("type_id", &ChestConfig::type_id)
      .def_readwrite("type_name", &ChestConfig::type_name)
      .def_readwrite("tag_ids", &ChestConfig::tag_ids)
      .def_readwrite("resource_type", &ChestConfig::resource_type)
      .def_readwrite("position_deltas", &ChestConfig::position_deltas)
      .def_readwrite("initial_inventory", &ChestConfig::initial_inventory)
      .def_readwrite("max_inventory", &ChestConfig::max_inventory);
}

#endif  // PACKAGES_METTAGRID_CPP_INCLUDE_METTAGRID_OBJECTS_CHEST_CONFIG_HPP_
