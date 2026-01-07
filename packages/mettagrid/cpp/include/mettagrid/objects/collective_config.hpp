#ifndef PACKAGES_METTAGRID_CPP_INCLUDE_METTAGRID_OBJECTS_COLLECTIVE_CONFIG_HPP_
#define PACKAGES_METTAGRID_CPP_INCLUDE_METTAGRID_OBJECTS_COLLECTIVE_CONFIG_HPP_

#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include <string>
#include <unordered_map>

#include "core/types.hpp"
#include "objects/inventory_config.hpp"

struct CollectiveConfig {
  std::string name;
  InventoryConfig inventory_config;
  std::unordered_map<InventoryItem, int> initial_inventory;

  CollectiveConfig() = default;
  explicit CollectiveConfig(const std::string& name) : name(name) {}
};

namespace py = pybind11;

inline void bind_collective_config(py::module& m) {
  py::class_<CollectiveConfig, std::shared_ptr<CollectiveConfig>>(m, "CollectiveConfig")
      .def(py::init<>())
      .def(py::init<const std::string&>(), py::arg("name"))
      .def_readwrite("name", &CollectiveConfig::name)
      .def_readwrite("inventory_config", &CollectiveConfig::inventory_config)
      .def_readwrite("initial_inventory", &CollectiveConfig::initial_inventory);
}

#endif  // PACKAGES_METTAGRID_CPP_INCLUDE_METTAGRID_OBJECTS_COLLECTIVE_CONFIG_HPP_
