#ifndef PACKAGES_METTAGRID_CPP_INCLUDE_METTAGRID_OBJECTS_INVENTORY_CONFIG_HPP_
#define PACKAGES_METTAGRID_CPP_INCLUDE_METTAGRID_OBJECTS_INVENTORY_CONFIG_HPP_

#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include <unordered_map>
#include <vector>

#include "core/types.hpp"
#include "objects/constants.hpp"

// Limit definition: resources that share the limit, base limit value, and modifiers
// Modifiers map: item_id -> bonus_per_item (e.g., gear -> 1 means each gear adds 1 to limit)
struct LimitDef {
  std::vector<InventoryItem> resources;
  InventoryQuantity base_limit;
  std::unordered_map<InventoryItem, InventoryQuantity> modifiers;

  LimitDef() : base_limit(0) {}
  LimitDef(const std::vector<InventoryItem>& resources,
           InventoryQuantity base_limit,
           const std::unordered_map<InventoryItem, InventoryQuantity>& modifiers = {})
      : resources(resources), base_limit(base_limit), modifiers(modifiers) {}
};

struct InventoryConfig {
  std::vector<LimitDef> limit_defs;

  InventoryConfig() = default;
};

namespace py = pybind11;

inline void bind_inventory_config(py::module& m) {
  py::class_<LimitDef>(m, "LimitDef")
      .def(py::init<>())
      .def(py::init<const std::vector<InventoryItem>&,
                    InventoryQuantity,
                    const std::unordered_map<InventoryItem, InventoryQuantity>&>(),
           py::arg("resources"),
           py::arg("base_limit"),
           py::arg("modifiers") = std::unordered_map<InventoryItem, InventoryQuantity>())
      .def_readwrite("resources", &LimitDef::resources)
      .def_readwrite("base_limit", &LimitDef::base_limit)
      .def_readwrite("modifiers", &LimitDef::modifiers);

  py::class_<InventoryConfig>(m, "InventoryConfig")
      .def(py::init<>())
      .def_readwrite("limit_defs", &InventoryConfig::limit_defs);
}
#endif  // PACKAGES_METTAGRID_CPP_INCLUDE_METTAGRID_OBJECTS_INVENTORY_CONFIG_HPP_
