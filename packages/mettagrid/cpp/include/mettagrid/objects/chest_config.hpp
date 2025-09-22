#ifndef PACKAGES_METTAGRID_CPP_INCLUDE_METTAGRID_OBJECTS_CHEST_CONFIG_HPP_
#define PACKAGES_METTAGRID_CPP_INCLUDE_METTAGRID_OBJECTS_CHEST_CONFIG_HPP_

#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include <set>

#include "core/grid_object.hpp"
#include "core/types.hpp"

struct ChestConfig : public GridObjectConfig {
  ChestConfig(TypeId type_id,
              const std::string& type_name,
              InventoryItem resource_type,
              const std::set<int>& deposit_positions,
              const std::set<int>& withdrawal_positions,
              const std::vector<int>& tag_ids = {})
      : GridObjectConfig(type_id, type_name, tag_ids),
        resource_type(resource_type),
        deposit_positions(deposit_positions),
        withdrawal_positions(withdrawal_positions) {}

  InventoryItem resource_type;
  std::set<int> deposit_positions;
  std::set<int> withdrawal_positions;
};

namespace py = pybind11;

inline void bind_chest_config(py::module& m) {
  py::class_<ChestConfig, GridObjectConfig, std::shared_ptr<ChestConfig>>(m, "ChestConfig")
      .def(py::init<TypeId, const std::string&, InventoryItem, const std::set<int>&, const std::set<int>&, const std::vector<int>&>(),
           py::arg("type_id"),
           py::arg("type_name"),
           py::arg("resource_type"),
           py::arg("deposit_positions"),
           py::arg("withdrawal_positions"),
           py::arg("tag_ids") = std::vector<int>())
      .def_readwrite("type_id", &ChestConfig::type_id)
      .def_readwrite("type_name", &ChestConfig::type_name)
      .def_readwrite("resource_type", &ChestConfig::resource_type)
      .def_readwrite("deposit_positions", &ChestConfig::deposit_positions)
      .def_readwrite("withdrawal_positions", &ChestConfig::withdrawal_positions)
      .def_readwrite("tag_ids", &ChestConfig::tag_ids);
}

#endif  // PACKAGES_METTAGRID_CPP_INCLUDE_METTAGRID_OBJECTS_CHEST_CONFIG_HPP_
