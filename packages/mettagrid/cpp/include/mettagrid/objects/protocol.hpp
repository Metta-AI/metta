#ifndef PACKAGES_METTAGRID_CPP_INCLUDE_METTAGRID_OBJECTS_PROTOCOL_HPP_
#define PACKAGES_METTAGRID_CPP_INCLUDE_METTAGRID_OBJECTS_PROTOCOL_HPP_

#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include <algorithm>
#include <cmath>
#include <memory>
#include <unordered_map>

#include "core/types.hpp"

class Protocol {
public:
  unsigned short min_agents;
  std::vector<ObservationType> vibes;
  std::unordered_map<InventoryItem, InventoryQuantity> input_resources;
  std::unordered_map<InventoryItem, InventoryQuantity> output_resources;
  unsigned short cooldown;
  float slope;     // Linear component: adds slope * activation_count to the multiplier
  float exponent;  // Exponential component: multiplies by (1 + exponent)^activation_count

  // Track activation count for inflation calculation
  mutable unsigned int activation_count;

  Protocol(unsigned short min_agents = 0,
           const std::vector<ObservationType>& vibes = {},
           const std::unordered_map<InventoryItem, InventoryQuantity>& inputs = {},
           const std::unordered_map<InventoryItem, InventoryQuantity>& outputs = {},
           unsigned short cooldown = 0,
           float slope = 0.0f,
           float exponent = 0.0f)
      : min_agents(min_agents),
        vibes(vibes),
        input_resources(inputs),
        output_resources(outputs),
        cooldown(cooldown),
        slope(slope),
        exponent(exponent),
        activation_count(0) {}

  // Calculate cost multiplier based on activation count
  // Formula: max(0, 1 + slope * n) * (1 + exponent)^n
  // - slope < 0: discounting (starts at 1, decreases linearly, floors at 0)
  // - slope > 0: linear cost increase
  // - exponent > 0: exponential cost increase
  // - exponent < 0: exponential cost decrease
  float get_cost_multiplier() const {
    float n = static_cast<float>(activation_count);
    float linear_component = std::max(0.0f, 1.0f + slope * n);
    float exponential_component = std::pow(1.0f + exponent, n);
    return linear_component * exponential_component;
  }
};

inline void bind_protocol(py::module& m) {
  py::class_<Protocol, std::shared_ptr<Protocol>>(m, "Protocol")
      .def(py::init())
      .def_readwrite("min_agents", &Protocol::min_agents)
      .def_readwrite("vibes", &Protocol::vibes)
      .def_readwrite("input_resources", &Protocol::input_resources)
      .def_readwrite("output_resources", &Protocol::output_resources)
      .def_readwrite("cooldown", &Protocol::cooldown)
      .def_readwrite("slope", &Protocol::slope)
      .def_readwrite("exponent", &Protocol::exponent)
      .def_readwrite("activation_count", &Protocol::activation_count);
}

#endif  // PACKAGES_METTAGRID_CPP_INCLUDE_METTAGRID_OBJECTS_PROTOCOL_HPP_
