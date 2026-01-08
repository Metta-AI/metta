#ifndef PACKAGES_METTAGRID_CPP_INCLUDE_METTAGRID_OBJECTS_FACTION_CONFIG_HPP_
#define PACKAGES_METTAGRID_CPP_INCLUDE_METTAGRID_OBJECTS_FACTION_CONFIG_HPP_

#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include <string>
#include <unordered_map>

#include "core/types.hpp"
#include "objects/inventory_config.hpp"

struct FactionConfig {
  std::string name;
  InventoryConfig inventory_config;
  std::unordered_map<InventoryItem, int> initial_inventory;

  FactionConfig() = default;
  explicit FactionConfig(const std::string& name) : name(name) {}
};

namespace py = pybind11;

inline void bind_faction_config(py::module& m) {
  py::class_<FactionConfig, std::shared_ptr<FactionConfig>>(m, "FactionConfig")
      .def(py::init<>())
      .def(py::init<const std::string&>(), py::arg("name"))
      .def_readwrite("name", &FactionConfig::name)
      .def_readwrite("inventory_config", &FactionConfig::inventory_config)
      .def_readwrite("initial_inventory", &FactionConfig::initial_inventory);
}

#endif  // PACKAGES_METTAGRID_CPP_INCLUDE_METTAGRID_OBJECTS_FACTION_CONFIG_HPP_
