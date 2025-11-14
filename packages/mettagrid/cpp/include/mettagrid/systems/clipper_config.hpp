#ifndef PACKAGES_METTAGRID_CPP_INCLUDE_METTAGRID_SYSTEMS_CLIPPER_CONFIG_HPP_
#define PACKAGES_METTAGRID_CPP_INCLUDE_METTAGRID_SYSTEMS_CLIPPER_CONFIG_HPP_

#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include <memory>

#include "objects/protocol.hpp"

struct ClipperConfig {
  std::vector<std::shared_ptr<Protocol>> unclipping_protocols;
  GridCoord length_scale;
  uint32_t scaled_cutoff_distance;
  uint32_t clip_period;

  ClipperConfig(std::vector<std::shared_ptr<Protocol>> protocol_ptrs = {},
                GridCoord length_scale = 0u,
                uint32_t scaled_cutoff_distance = 3,
                uint32_t clip_period = 0)
      : unclipping_protocols(std::move(protocol_ptrs)),
        length_scale(length_scale),
        scaled_cutoff_distance(scaled_cutoff_distance),
        clip_period(clip_period) {}
};

namespace py = pybind11;

inline void bind_clipper_config(py::module& m) {
  py::class_<ClipperConfig, std::shared_ptr<ClipperConfig>>(m, "ClipperConfig")
      .def(py::init<>())
      .def_readwrite("unclipping_protocols", &ClipperConfig::unclipping_protocols)
      .def_readwrite("length_scale", &ClipperConfig::length_scale)
      .def_readwrite("scaled_cutoff_distance", &ClipperConfig::scaled_cutoff_distance)
      .def_readwrite("clip_period", &ClipperConfig::clip_period);
}

#endif  // PACKAGES_METTAGRID_CPP_INCLUDE_METTAGRID_SYSTEMS_CLIPPER_CONFIG_HPP_
