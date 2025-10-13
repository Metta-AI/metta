// assembler_config.hpp
#ifndef PACKAGES_METTAGRID_CPP_INCLUDE_METTAGRID_OBJECTS_ASSEMBLER_CONFIG_HPP_
#define PACKAGES_METTAGRID_CPP_INCLUDE_METTAGRID_OBJECTS_ASSEMBLER_CONFIG_HPP_

#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include <string>
#include <unordered_map>
#include <vector>

#include "core/grid_object.hpp"
#include "core/types.hpp"
#include "objects/recipe.hpp"

struct AssemblerConfig : public GridObjectConfig {
  AssemblerConfig(TypeId type_id, const std::string& type_name, const std::vector<int>& tag_ids = {})
      : GridObjectConfig(type_id, type_name, tag_ids),
        recipe_details_obs(false),
        input_recipe_offset(0),
        output_recipe_offset(0),
        allow_partial_usage(false),
        max_uses(0),             // 0 means unlimited uses
        exhaustion(0.0f),        // 0 means no exhaustion
        clip_immune(false),      // Not immune by default
        start_clipped(false) {}  // Not clipped at start by default

  // Recipes will be set separately via initialize_recipes()
  std::vector<std::shared_ptr<Recipe>> recipes;

  // Recipe observation configuration
  bool recipe_details_obs;
  ObservationType input_recipe_offset;
  ObservationType output_recipe_offset;

  // Allow partial usage during cooldown
  bool allow_partial_usage;
  // Maximum number of uses (0 = unlimited)
  unsigned int max_uses;

  // Exhaustion rate - cooldown multiplier grows by (1 + exhaustion) each use
  float exhaustion;

  // Clip immunity - if true, this assembler cannot be clipped
  bool clip_immune;

  // Start clipped - if true, this assembler starts in a clipped state
  bool start_clipped;
};

namespace py = pybind11;

inline void bind_assembler_config(py::module& m) {
  py::class_<AssemblerConfig, GridObjectConfig, std::shared_ptr<AssemblerConfig>>(m, "AssemblerConfig")
      .def(py::init<TypeId, const std::string&, const std::vector<int>&>(),
           py::arg("type_id"),
           py::arg("type_name"),
           py::arg("tag_ids") = std::vector<int>{})
      .def_readwrite("type_id", &AssemblerConfig::type_id)
      .def_readwrite("type_name", &AssemblerConfig::type_name)
      .def_readwrite("tag_ids", &AssemblerConfig::tag_ids)
      .def_readwrite("recipes", &AssemblerConfig::recipes)
      .def_readwrite("recipe_details_obs", &AssemblerConfig::recipe_details_obs)
      .def_readwrite("allow_partial_usage", &AssemblerConfig::allow_partial_usage)
      .def_readwrite("max_uses", &AssemblerConfig::max_uses)
      .def_readwrite("exhaustion", &AssemblerConfig::exhaustion)
      .def_readwrite("clip_immune", &AssemblerConfig::clip_immune)
      .def_readwrite("start_clipped", &AssemblerConfig::start_clipped);
}

#endif  // PACKAGES_METTAGRID_CPP_INCLUDE_METTAGRID_OBJECTS_ASSEMBLER_CONFIG_HPP_
