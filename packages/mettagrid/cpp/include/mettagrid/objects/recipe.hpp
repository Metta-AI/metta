#ifndef PACKAGES_METTAGRID_CPP_INCLUDE_METTAGRID_OBJECTS_RECIPE_HPP_
#define PACKAGES_METTAGRID_CPP_INCLUDE_METTAGRID_OBJECTS_RECIPE_HPP_

#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include <memory>
#include <unordered_map>

#include "core/types.hpp"

// Recipe class for Assembler
class Recipe {
public:
  std::unordered_map<InventoryItem, InventoryQuantity> input_resources;
  std::unordered_map<InventoryItem, InventoryQuantity> output_resources;
  unsigned short cooldown;

  Recipe(const std::unordered_map<InventoryItem, InventoryQuantity>& inputs = {},
         const std::unordered_map<InventoryItem, InventoryQuantity>& outputs = {},
         unsigned short cooldown = 0)
      : input_resources(inputs), output_resources(outputs), cooldown(cooldown) {}
};

inline void bind_recipe(py::module& m) {
  py::class_<Recipe, std::shared_ptr<Recipe>>(m, "Recipe")
      .def(py::init<const std::unordered_map<InventoryItem, InventoryQuantity>&,
                    const std::unordered_map<InventoryItem, InventoryQuantity>&,
                    unsigned short>(),
           py::arg("input_resources") = std::unordered_map<InventoryItem, InventoryQuantity>(),
           py::arg("output_resources") = std::unordered_map<InventoryItem, InventoryQuantity>(),
           py::arg("cooldown") = 0)
      .def_readwrite("input_resources", &Recipe::input_resources)
      .def_readwrite("output_resources", &Recipe::output_resources)
      .def_readwrite("cooldown", &Recipe::cooldown);
}

#endif  // PACKAGES_METTAGRID_CPP_INCLUDE_METTAGRID_OBJECTS_RECIPE_HPP_
