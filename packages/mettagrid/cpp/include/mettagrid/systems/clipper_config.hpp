#ifndef PACKAGES_METTAGRID_CPP_INCLUDE_METTAGRID_SYSTEMS_CLIPPER_CONFIG_HPP_
#define PACKAGES_METTAGRID_CPP_INCLUDE_METTAGRID_SYSTEMS_CLIPPER_CONFIG_HPP_

#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include <memory>

#include "objects/recipe.hpp"

struct ClipperConfig {
  std::shared_ptr<Recipe> recipe;
  float length_scale;
  float cutoff_distance;
  float clip_rate;
  bool auto_length_scale;
  float length_scale_factor;

  ClipperConfig(std::shared_ptr<Recipe> recipe_ptr, float length_scale, float cutoff_distance, float clip_rate,
                bool auto_length_scale = true, float length_scale_factor = 1.0f)
      : recipe(std::move(recipe_ptr)),
        length_scale(length_scale),
        cutoff_distance(cutoff_distance),
        clip_rate(clip_rate),
        auto_length_scale(auto_length_scale),
        length_scale_factor(length_scale_factor) {}
};

namespace py = pybind11;

inline void bind_clipper_config(py::module& m) {
  py::class_<ClipperConfig, std::shared_ptr<ClipperConfig>>(m, "ClipperConfig")
      .def(py::init<std::shared_ptr<Recipe>, float, float, float, bool, float>(),
           py::arg("recipe"),
           py::arg("length_scale"),
           py::arg("cutoff_distance"),
           py::arg("clip_rate"),
           py::arg("auto_length_scale") = true,
           py::arg("length_scale_factor") = 1.0f)
      .def_readwrite("recipe", &ClipperConfig::recipe)
      .def_readwrite("length_scale", &ClipperConfig::length_scale)
      .def_readwrite("cutoff_distance", &ClipperConfig::cutoff_distance)
      .def_readwrite("clip_rate", &ClipperConfig::clip_rate)
      .def_readwrite("auto_length_scale", &ClipperConfig::auto_length_scale)
      .def_readwrite("length_scale_factor", &ClipperConfig::length_scale_factor);
}

#endif  // PACKAGES_METTAGRID_CPP_INCLUDE_METTAGRID_SYSTEMS_CLIPPER_CONFIG_HPP_
