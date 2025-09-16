#ifndef OBJECTS_GROUP_CONVERTER_CONFIG_HPP_
#define OBJECTS_GROUP_CONVERTER_CONFIG_HPP_

#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include <map>
#include <string>

#include "grid_object.hpp"
#include "types.hpp"

struct GroupConverterRecipe {
  std::map<InventoryItem, InventoryQuantity> input_resources;
  std::map<InventoryItem, InventoryQuantity> output_resources;
  uint16_t discovery_bonus;

  GroupConverterRecipe(
      const std::map<InventoryItem, InventoryQuantity>& inputs,
      const std::map<InventoryItem, InventoryQuantity>& outputs,
      uint16_t bonus = 0)
      : input_resources(inputs), output_resources(outputs), discovery_bonus(bonus) {}
};

struct GroupConverterConfig : public GridObjectConfig {
  GroupConverterConfig(TypeId type_id,
                      const std::string& type_name,
                      short max_output = -1,
                      short max_conversions = -1,
                      unsigned short conversion_ticks = 10,
                      unsigned short cooldown = 0,
                      ObservationType color = 0)
      : GridObjectConfig(type_id, type_name),
        max_output(max_output),
        max_conversions(max_conversions),
        conversion_ticks(conversion_ticks),
        cooldown(cooldown),
        color(color) {}

  short max_output;
  short max_conversions;
  unsigned short conversion_ticks;
  unsigned short cooldown;
  ObservationType color;
  std::map<uint8_t, GroupConverterRecipe> recipes;
};

namespace py = pybind11;

inline void bind_group_converter_config(py::module& m) {
  py::class_<GroupConverterRecipe>(m, "GroupConverterRecipe")
      .def(py::init<const std::map<InventoryItem, InventoryQuantity>&,
                    const std::map<InventoryItem, InventoryQuantity>&,
                    uint16_t>(),
           py::arg("input_resources"),
           py::arg("output_resources"),
           py::arg("discovery_bonus") = 0)
      .def_readwrite("input_resources", &GroupConverterRecipe::input_resources)
      .def_readwrite("output_resources", &GroupConverterRecipe::output_resources)
      .def_readwrite("discovery_bonus", &GroupConverterRecipe::discovery_bonus);

  py::class_<GroupConverterConfig, GridObjectConfig, std::shared_ptr<GroupConverterConfig>>(m, "GroupConverterConfig")
      .def(py::init<TypeId,
                    const std::string&,
                    short,
                    short,
                    unsigned short,
                    unsigned short,
                    ObservationType>(),
           py::arg("type_id"),
           py::arg("type_name"),
           py::arg("max_output") = -1,
           py::arg("max_conversions") = -1,
           py::arg("conversion_ticks") = 10,
           py::arg("cooldown") = 0,
           py::arg("color") = 0)
      .def_readwrite("type_id", &GroupConverterConfig::type_id)
      .def_readwrite("type_name", &GroupConverterConfig::type_name)
      .def_readwrite("max_output", &GroupConverterConfig::max_output)
      .def_readwrite("max_conversions", &GroupConverterConfig::max_conversions)
      .def_readwrite("conversion_ticks", &GroupConverterConfig::conversion_ticks)
      .def_readwrite("cooldown", &GroupConverterConfig::cooldown)
      .def_readwrite("color", &GroupConverterConfig::color)
      .def_readwrite("recipes", &GroupConverterConfig::recipes);
}

#endif  // OBJECTS_GROUP_CONVERTER_CONFIG_HPP_