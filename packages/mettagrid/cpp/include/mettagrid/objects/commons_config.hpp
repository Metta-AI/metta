#ifndef PACKAGES_METTAGRID_CPP_INCLUDE_METTAGRID_OBJECTS_COMMONS_CONFIG_HPP_
#define PACKAGES_METTAGRID_CPP_INCLUDE_METTAGRID_OBJECTS_COMMONS_CONFIG_HPP_

#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include <string>
#include <unordered_map>

#include "core/types.hpp"
#include "objects/inventory_config.hpp"

struct CommonsConfig {
  std::string name;
  InventoryConfig inventory_config;
  std::unordered_map<InventoryItem, int> initial_inventory;

  CommonsConfig() = default;
  explicit CommonsConfig(const std::string& name) : name(name) {}
};

namespace py = pybind11;

inline void bind_commons_config(py::module& m) {
  py::class_<CommonsConfig, std::shared_ptr<CommonsConfig>>(m, "CommonsConfig")
      .def(py::init<>())
      .def(py::init<const std::string&>(), py::arg("name"))
      .def_readwrite("name", &CommonsConfig::name)
      .def_readwrite("inventory_config", &CommonsConfig::inventory_config)
      .def_readwrite("initial_inventory", &CommonsConfig::initial_inventory);
}

#endif  // PACKAGES_METTAGRID_CPP_INCLUDE_METTAGRID_OBJECTS_COMMONS_CONFIG_HPP_
