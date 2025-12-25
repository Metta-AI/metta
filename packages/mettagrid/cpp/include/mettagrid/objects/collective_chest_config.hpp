#ifndef PACKAGES_METTAGRID_CPP_INCLUDE_METTAGRID_OBJECTS_COLLECTIVE_CHEST_CONFIG_HPP_
#define PACKAGES_METTAGRID_CPP_INCLUDE_METTAGRID_OBJECTS_COLLECTIVE_CHEST_CONFIG_HPP_

#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include "objects/chest_config.hpp"

// CollectiveChestConfig extends ChestConfig - uses the same config but creates a CollectiveChest
struct CollectiveChestConfig : public ChestConfig {
  CollectiveChestConfig(TypeId type_id, const std::string& type_name, ObservationType initial_vibe = 0)
      : ChestConfig(type_id, type_name, initial_vibe) {}
};

namespace py = pybind11;

inline void bind_collective_chest_config(py::module& m) {
  py::class_<CollectiveChestConfig, ChestConfig, std::shared_ptr<CollectiveChestConfig>>(m, "CollectiveChestConfig")
      .def(py::init<TypeId, const std::string&, ObservationType>(),
           py::arg("type_id"),
           py::arg("type_name"),
           py::arg("initial_vibe") = 0);
}

#endif  // PACKAGES_METTAGRID_CPP_INCLUDE_METTAGRID_OBJECTS_COLLECTIVE_CHEST_CONFIG_HPP_
