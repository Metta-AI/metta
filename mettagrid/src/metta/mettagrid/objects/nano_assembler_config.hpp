// nano_assembler_config.hpp
#ifndef OBJECTS_NANO_ASSEMBLER_CONFIG_HPP_
#define OBJECTS_NANO_ASSEMBLER_CONFIG_HPP_

#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include <map>
#include <string>
#include <vector>

#include "grid_object.hpp"
#include "recipe.hpp"
#include "types.hpp"

struct NanoAssemblerConfig : public GridObjectConfig {
  NanoAssemblerConfig(TypeId type_id, const std::string& type_name) : GridObjectConfig(type_id, type_name) {}

  // Recipes will be set separately via initialize_recipes()
  std::vector<Recipe*> recipes;
};

namespace py = pybind11;

inline void bind_nano_assembler_config(py::module& m) {
  py::class_<NanoAssemblerConfig, GridObjectConfig, std::shared_ptr<NanoAssemblerConfig>>(m, "NanoAssemblerConfig")
      .def(py::init<TypeId, const std::string&>(), py::arg("type_id"), py::arg("type_name"))
      .def_readwrite("type_id", &NanoAssemblerConfig::type_id)
      .def_readwrite("type_name", &NanoAssemblerConfig::type_name)
      .def_readwrite("recipes", &NanoAssemblerConfig::recipes);
}

#endif  // OBJECTS_NANO_ASSEMBLER_CONFIG_HPP_
