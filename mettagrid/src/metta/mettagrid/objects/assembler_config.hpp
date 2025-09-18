// assembler_config.hpp
#ifndef OBJECTS_ASSEMBLER_CONFIG_HPP_
#define OBJECTS_ASSEMBLER_CONFIG_HPP_

#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include <map>
#include <string>
#include <vector>

#include "grid_object.hpp"
#include "recipe.hpp"
#include "types.hpp"

struct AssemblerConfig : public GridObjectConfig {
  AssemblerConfig(TypeId type_id, const std::string& type_name) : GridObjectConfig(type_id, type_name) {}

  // Recipes will be set separately via initialize_recipes()
  std::vector<std::shared_ptr<Recipe>> recipes;
};

namespace py = pybind11;

inline void bind_assembler_config(py::module& m) {
  py::class_<AssemblerConfig, GridObjectConfig, std::shared_ptr<AssemblerConfig>>(m, "AssemblerConfig")
      .def(py::init<TypeId, const std::string&>(), py::arg("type_id"), py::arg("type_name"))
      .def_readwrite("type_id", &AssemblerConfig::type_id)
      .def_readwrite("type_name", &AssemblerConfig::type_name)
      .def_readwrite("recipes", &AssemblerConfig::recipes);
}

#endif  // OBJECTS_ASSEMBLER_CONFIG_HPP_
