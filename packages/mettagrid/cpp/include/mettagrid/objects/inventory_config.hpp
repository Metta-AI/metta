#ifndef PACKAGES_METTAGRID_CPP_INCLUDE_METTAGRID_OBJECTS_INVENTORY_CONFIG_HPP_
#define PACKAGES_METTAGRID_CPP_INCLUDE_METTAGRID_OBJECTS_INVENTORY_CONFIG_HPP_

#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include <vector>

#include "core/types.hpp"
#include "objects/constants.hpp"

struct InventoryConfig {
  std::vector<std::pair<std::vector<InventoryItem>, InventoryQuantity>> limits;

  InventoryConfig() = default;

  explicit InventoryConfig(const std::vector<std::pair<std::vector<InventoryItem>, InventoryQuantity>>& limits)
      : limits(limits) {}
};

namespace py = pybind11;

inline void bind_inventory_config(py::module& m) {
  py::class_<InventoryConfig>(m, "InventoryConfig")
      .def(py::init<>())
      .def(py::init<const std::vector<std::pair<std::vector<InventoryItem>, InventoryQuantity>>&>(),
           py::arg("limits") = std::vector<std::pair<std::vector<InventoryItem>, InventoryQuantity>>())
      .def_readwrite("limits", &InventoryConfig::limits);
}
#endif  // PACKAGES_METTAGRID_CPP_INCLUDE_METTAGRID_OBJECTS_INVENTORY_CONFIG_HPP_
