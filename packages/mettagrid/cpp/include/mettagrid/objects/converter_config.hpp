// converter_config.hpp
#ifndef PACKAGES_METTAGRID_CPP_INCLUDE_METTAGRID_OBJECTS_CONVERTER_CONFIG_HPP_
#define PACKAGES_METTAGRID_CPP_INCLUDE_METTAGRID_OBJECTS_CONVERTER_CONFIG_HPP_

#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include <map>
#include <string>

#include "core/grid_object.hpp"
#include "core/types.hpp"

struct ConverterConfig : public GridObjectConfig {
  ConverterConfig(TypeId type_id,
                  const std::string& type_name,
                  const std::map<InventoryItem, InventoryQuantity>& input_resources,
                  const std::map<InventoryItem, InventoryQuantity>& output_resources,
                  short max_output,
                  short max_conversions,
                  unsigned short conversion_ticks,
                  unsigned short cooldown,
                  InventoryQuantity initial_resource_count = 0,
                  ObservationType color = 0,
                  bool recipe_details_obs = false,
                  const std::vector<int>& tag_ids = {})
      : GridObjectConfig(type_id, type_name, tag_ids),
        input_resources(input_resources),
        output_resources(output_resources),
        max_output(max_output),
        max_conversions(max_conversions),
        conversion_ticks(conversion_ticks),
        cooldown(cooldown),
        initial_resource_count(initial_resource_count),
        color(color),
        recipe_details_obs(recipe_details_obs),
        input_recipe_offset(0),
        output_recipe_offset(0) {}

  std::map<InventoryItem, InventoryQuantity> input_resources;
  std::map<InventoryItem, InventoryQuantity> output_resources;
  short max_output;
  short max_conversions;
  unsigned short conversion_ticks;
  unsigned short cooldown;
  InventoryQuantity initial_resource_count;
  ObservationType color;
  bool recipe_details_obs;
  ObservationType input_recipe_offset;
  ObservationType output_recipe_offset;
};

namespace py = pybind11;

inline void bind_converter_config(py::module& m) {
  py::class_<ConverterConfig, GridObjectConfig, std::shared_ptr<ConverterConfig>>(m, "ConverterConfig")
      .def(py::init<TypeId,
                    const std::string&,
                    const std::map<InventoryItem, InventoryQuantity>&,
                    const std::map<InventoryItem, InventoryQuantity>&,
                    short,
                    short,
                    unsigned short,
                    unsigned short,
                    unsigned char,
                    ObservationType,
                    bool,
                    const std::vector<int>&>(),
           py::arg("type_id"),
           py::arg("type_name"),
           py::arg("input_resources"),
           py::arg("output_resources"),
           py::arg("max_output"),
           py::arg("max_conversions"),
           py::arg("conversion_ticks"),
           py::arg("cooldown"),
           py::arg("initial_resource_count") = 0,
           py::arg("color") = 0,
           py::arg("recipe_details_obs") = false,
           py::arg("tag_ids") = std::vector<int>())
      .def_readwrite("type_id", &ConverterConfig::type_id)
      .def_readwrite("type_name", &ConverterConfig::type_name)
      .def_readwrite("input_resources", &ConverterConfig::input_resources)
      .def_readwrite("output_resources", &ConverterConfig::output_resources)
      .def_readwrite("max_output", &ConverterConfig::max_output)
      .def_readwrite("max_conversions", &ConverterConfig::max_conversions)
      .def_readwrite("conversion_ticks", &ConverterConfig::conversion_ticks)
      .def_readwrite("cooldown", &ConverterConfig::cooldown)
      .def_readwrite("initial_resource_count", &ConverterConfig::initial_resource_count)
      .def_readwrite("color", &ConverterConfig::color)
      .def_readwrite("recipe_details_obs", &ConverterConfig::recipe_details_obs)
      .def_readwrite("tag_ids", &ConverterConfig::tag_ids);
}

#endif  // PACKAGES_METTAGRID_CPP_INCLUDE_METTAGRID_OBJECTS_CONVERTER_CONFIG_HPP_
