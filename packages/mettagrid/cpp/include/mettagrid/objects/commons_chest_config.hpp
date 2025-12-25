#ifndef PACKAGES_METTAGRID_CPP_INCLUDE_METTAGRID_OBJECTS_COMMONS_CHEST_CONFIG_HPP_
#define PACKAGES_METTAGRID_CPP_INCLUDE_METTAGRID_OBJECTS_COMMONS_CHEST_CONFIG_HPP_

#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include "objects/chest_config.hpp"

// CommonsChestConfig extends ChestConfig - uses the same config but creates a CommonsChest
struct CommonsChestConfig : public ChestConfig {
  CommonsChestConfig(TypeId type_id, const std::string& type_name, ObservationType initial_vibe = 0)
      : ChestConfig(type_id, type_name, initial_vibe) {}
};

namespace py = pybind11;

inline void bind_commons_chest_config(py::module& m) {
  py::class_<CommonsChestConfig, ChestConfig, std::shared_ptr<CommonsChestConfig>>(m, "CommonsChestConfig")
      .def(py::init<TypeId, const std::string&, ObservationType>(),
           py::arg("type_id"),
           py::arg("type_name"),
           py::arg("initial_vibe") = 0);
}

#endif  // PACKAGES_METTAGRID_CPP_INCLUDE_METTAGRID_OBJECTS_COMMONS_CHEST_CONFIG_HPP_
