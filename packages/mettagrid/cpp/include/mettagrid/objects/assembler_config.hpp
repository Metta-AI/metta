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
#include "objects/protocol.hpp"

struct AssemblerConfig : public GridObjectConfig {
  AssemblerConfig(TypeId type_id, const std::string& type_name, ObservationType initial_vibe = 0)
      : GridObjectConfig(type_id, type_name, initial_vibe),
        allow_partial_usage(false),
        max_uses(0),                 // 0 means unlimited uses
        clip_immune(false),          // Not immune by default
        start_clipped(false),        // Not clipped at start by default
        chest_search_distance(0) {}  // 0 means chests are not searched

  // List of protocols - GroupVibe keys will be calculated from each protocol's vibes vector
  std::vector<std::shared_ptr<Protocol>> protocols;

  // Allow partial usage during cooldown
  bool allow_partial_usage;
  // Maximum number of uses (0 = unlimited)
  unsigned int max_uses;

  // Clip immunity - if true, this assembler cannot be clipped
  bool clip_immune;

  // Start clipped - if true, this assembler starts in a clipped state
  bool start_clipped;

  // Chest search distance - if > 0, assembler can use inventories from chests within this distance
  // Distance is measured as Chebyshev distance (max of row and column differences)
  // 0 means chests are not searched
  unsigned int chest_search_distance;
};

namespace py = pybind11;

inline void bind_assembler_config(py::module& m) {
  py::class_<AssemblerConfig, GridObjectConfig, std::shared_ptr<AssemblerConfig>>(m, "AssemblerConfig")
      .def(py::init<TypeId, const std::string&, ObservationType>(),
           py::arg("type_id"),
           py::arg("type_name"),
           py::arg("initial_vibe") = 0)
      .def_readwrite("type_id", &AssemblerConfig::type_id)
      .def_readwrite("type_name", &AssemblerConfig::type_name)
      .def_readwrite("tag_ids", &AssemblerConfig::tag_ids)
      .def_readwrite("protocols", &AssemblerConfig::protocols)
      .def_readwrite("allow_partial_usage", &AssemblerConfig::allow_partial_usage)
      .def_readwrite("max_uses", &AssemblerConfig::max_uses)
      .def_readwrite("clip_immune", &AssemblerConfig::clip_immune)
      .def_readwrite("start_clipped", &AssemblerConfig::start_clipped)
      .def_readwrite("chest_search_distance", &AssemblerConfig::chest_search_distance)
      .def_readwrite("initial_vibe", &AssemblerConfig::initial_vibe);
}

#endif  // PACKAGES_METTAGRID_CPP_INCLUDE_METTAGRID_OBJECTS_ASSEMBLER_CONFIG_HPP_
