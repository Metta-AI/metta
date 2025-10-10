#ifndef PACKAGES_METTAGRID_CPP_INCLUDE_METTAGRID_SYSTEMS_CLIPPER_CONFIG_HPP_
#define PACKAGES_METTAGRID_CPP_INCLUDE_METTAGRID_SYSTEMS_CLIPPER_CONFIG_HPP_

#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include <memory>

#include "objects/recipe.hpp"

struct ClipperConfig {
  std::vector<std::shared_ptr<Recipe>> unclipping_recipes;
  float length_scale;
  float cutoff_distance;
  float clip_rate;

  ClipperConfig(std::vector<std::shared_ptr<Recipe>> recipe_ptrs, float length_scale, float cutoff_distance, float clip_rate)
      : unclipping_recipes(std::move(recipe_ptrs)),
        length_scale(length_scale),
        cutoff_distance(cutoff_distance),
        clip_rate(clip_rate) {}
};

namespace py = pybind11;

inline void bind_clipper_config(py::module& m) {
  py::class_<ClipperConfig, std::shared_ptr<ClipperConfig>>(m, "ClipperConfig")
      .def(py::init<std::vector<std::shared_ptr<Recipe>>, float, float, float>(),
           py::arg("unclipping_recipes"),
           py::arg("length_scale"),
           py::arg("cutoff_distance"),
           py::arg("clip_rate"))
      .def_readwrite("unclipping_recipes", &ClipperConfig::unclipping_recipes)
      .def_readwrite("length_scale", &ClipperConfig::length_scale)
      .def_readwrite("cutoff_distance", &ClipperConfig::cutoff_distance)
      .def_readwrite("clip_rate", &ClipperConfig::clip_rate);
}

#endif  // PACKAGES_METTAGRID_CPP_INCLUDE_METTAGRID_SYSTEMS_CLIPPER_CONFIG_HPP_
