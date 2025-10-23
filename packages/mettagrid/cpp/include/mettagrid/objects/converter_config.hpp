// converter_config.hpp
#ifndef PACKAGES_METTAGRID_CPP_INCLUDE_METTAGRID_OBJECTS_CONVERTER_CONFIG_HPP_
#define PACKAGES_METTAGRID_CPP_INCLUDE_METTAGRID_OBJECTS_CONVERTER_CONFIG_HPP_

#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include <string>
#include <unordered_map>
#include <vector>

#include "core/grid_object.hpp"
#include "core/types.hpp"

struct ConverterConfig : public GridObjectConfig {
  ConverterConfig(TypeId type_id,
                  const std::string& type_name,
                  const std::unordered_map<InventoryItem, InventoryQuantity>& input_resources,
                  const std::unordered_map<InventoryItem, InventoryQuantity>& output_resources,
                  short max_output,
                  short max_conversions,
                  unsigned short conversion_ticks,
                  const std::vector<unsigned short>& cooldown_schedule,
                  InventoryQuantity initial_resource_count = 0,
                  bool recipe_details_obs = false,
                  const std::vector<int>& tag_ids = {})
      : GridObjectConfig(type_id, type_name, tag_ids),
        input_resources(input_resources),
        output_resources(output_resources),
        max_output(max_output),
        max_conversions(max_conversions),
        conversion_ticks(conversion_ticks),
        cooldown_time(normalize_cooldown(cooldown_schedule)),
        initial_resource_count(initial_resource_count),
        recipe_details_obs(recipe_details_obs),
        input_recipe_offset(0),
        output_recipe_offset(0) {}

  std::unordered_map<InventoryItem, InventoryQuantity> input_resources;
  std::unordered_map<InventoryItem, InventoryQuantity> output_resources;
  short max_output;
  short max_conversions;
  unsigned short conversion_ticks;
  std::vector<unsigned short> cooldown_time;
  InventoryQuantity initial_resource_count;
  bool recipe_details_obs;
  ObservationType input_recipe_offset;
  ObservationType output_recipe_offset;

private:
  static std::vector<unsigned short> normalize_cooldown(const std::vector<unsigned short>& values) {
    if (values.empty()) {
      return std::vector<unsigned short>{0};
    }
    return values;
  }
};

namespace py = pybind11;

inline void bind_converter_config(py::module& m) {
  py::class_<ConverterConfig, GridObjectConfig, std::shared_ptr<ConverterConfig>>(m, "ConverterConfig")
      .def(py::init<TypeId,
                    const std::string&,
                    const std::unordered_map<InventoryItem, InventoryQuantity>&,
                    const std::unordered_map<InventoryItem, InventoryQuantity>&,
                    short,
                    short,
                    unsigned short,
                    const std::vector<unsigned short>&,
                    unsigned char,
                    bool,
                    const std::vector<int>&>(),
           py::arg("type_id"),
           py::arg("type_name"),
           py::arg("input_resources"),
           py::arg("output_resources"),
           py::arg("max_output"),
           py::arg("max_conversions"),
           py::arg("conversion_ticks"),
           py::arg("cooldown_time"),
           py::arg("initial_resource_count") = 0,
           py::arg("recipe_details_obs") = false,
           py::arg("tag_ids") = std::vector<int>())
      .def_readwrite("type_id", &ConverterConfig::type_id)
      .def_readwrite("type_name", &ConverterConfig::type_name)
      .def_readwrite("input_resources", &ConverterConfig::input_resources)
      .def_readwrite("output_resources", &ConverterConfig::output_resources)
      .def_readwrite("max_output", &ConverterConfig::max_output)
      .def_readwrite("max_conversions", &ConverterConfig::max_conversions)
      .def_readwrite("conversion_ticks", &ConverterConfig::conversion_ticks)
      .def_readwrite("cooldown_time", &ConverterConfig::cooldown_time)
      .def_readwrite("initial_resource_count", &ConverterConfig::initial_resource_count)
      .def_readwrite("recipe_details_obs", &ConverterConfig::recipe_details_obs)
      .def_readwrite("tag_ids", &ConverterConfig::tag_ids);
}

#endif  // PACKAGES_METTAGRID_CPP_INCLUDE_METTAGRID_OBJECTS_CONVERTER_CONFIG_HPP_
