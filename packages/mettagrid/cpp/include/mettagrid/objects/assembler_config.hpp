// assembler_config.hpp
#ifndef PACKAGES_METTAGRID_CPP_INCLUDE_METTAGRID_OBJECTS_ASSEMBLER_CONFIG_HPP_
#define PACKAGES_METTAGRID_CPP_INCLUDE_METTAGRID_OBJECTS_ASSEMBLER_CONFIG_HPP_

#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include <map>
#include <string>
#include <vector>

#include "core/grid_object.hpp"
#include "objects/recipe.hpp"
#include "core/types.hpp"

struct AssemblerConfig : public GridObjectConfig {
  AssemblerConfig(TypeId type_id, const std::string& type_name, const std::vector<int>& tag_ids = {})
    : GridObjectConfig(type_id, type_name, tag_ids) {}

  // Recipes will be set separately via initialize_recipes()
  std::vector<std::shared_ptr<Recipe>> recipes;
};

namespace py = pybind11;

inline void bind_assembler_config(py::module& m) {
  py::class_<AssemblerConfig, GridObjectConfig, std::shared_ptr<AssemblerConfig>>(m, "AssemblerConfig")
      .def(py::init<TypeId, const std::string&, const std::vector<int>&>(),
           py::arg("type_id"), py::arg("type_name"), py::arg("tag_ids") = std::vector<int>{})
      .def_readwrite("type_id", &AssemblerConfig::type_id)
      .def_readwrite("type_name", &AssemblerConfig::type_name)
      .def_readwrite("tag_ids", &AssemblerConfig::tag_ids)
      .def_readwrite("recipes", &AssemblerConfig::recipes);
}

#endif  // PACKAGES_METTAGRID_CPP_INCLUDE_METTAGRID_OBJECTS_ASSEMBLER_CONFIG_HPP_
