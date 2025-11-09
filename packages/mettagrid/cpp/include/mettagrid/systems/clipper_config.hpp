#ifndef PACKAGES_METTAGRID_CPP_INCLUDE_METTAGRID_SYSTEMS_CLIPPER_CONFIG_HPP_
#define PACKAGES_METTAGRID_CPP_INCLUDE_METTAGRID_SYSTEMS_CLIPPER_CONFIG_HPP_

#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include <memory>

#include "objects/protocol.hpp"

struct ClipperConfig {
  std::vector<std::shared_ptr<Protocol>> unclipping_protocols;
  float length_scale;
  float cutoff_distance;
  uint32_t clip_period;

  ClipperConfig(std::vector<std::shared_ptr<Protocol>> protocol_ptrs,
                float length_scale,
                float cutoff_distance,
                uint32_t clip_period)
      : unclipping_protocols(std::move(protocol_ptrs)),
        length_scale(length_scale),
        cutoff_distance(cutoff_distance),
        clip_period(clip_period) {}
};

namespace py = pybind11;

inline void bind_clipper_config(py::module& m) {
  py::class_<ClipperConfig, std::shared_ptr<ClipperConfig>>(m, "ClipperConfig")
      .def(py::init<std::vector<std::shared_ptr<Protocol>>, float, float, uint32_t>(),
           py::arg("unclipping_protocols"),
           py::arg("length_scale"),
           py::arg("cutoff_distance"),
           py::arg("clip_period"))
      .def_readwrite("unclipping_protocols", &ClipperConfig::unclipping_protocols)
      .def_readwrite("length_scale", &ClipperConfig::length_scale)
      .def_readwrite("cutoff_distance", &ClipperConfig::cutoff_distance)
      .def_readwrite("clip_period", &ClipperConfig::clip_period);
}

#endif  // PACKAGES_METTAGRID_CPP_INCLUDE_METTAGRID_SYSTEMS_CLIPPER_CONFIG_HPP_
