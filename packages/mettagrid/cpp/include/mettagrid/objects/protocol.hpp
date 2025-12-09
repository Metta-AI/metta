#ifndef PACKAGES_METTAGRID_CPP_INCLUDE_METTAGRID_OBJECTS_PROTOCOL_HPP_
#define PACKAGES_METTAGRID_CPP_INCLUDE_METTAGRID_OBJECTS_PROTOCOL_HPP_

#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

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
  unsigned int sigmoid;  // Number of discounted uses before full price
  float inflation;       // Compound rate for exponential cost scaling after sigmoid uses

  // Track activation count for cost multiplier calculation
  mutable unsigned int activation_count;

  Protocol(unsigned short min_agents = 0,
           const std::vector<ObservationType>& vibes = {},
           const std::unordered_map<InventoryItem, InventoryQuantity>& inputs = {},
           const std::unordered_map<InventoryItem, InventoryQuantity>& outputs = {},
           unsigned short cooldown = 0,
           unsigned int sigmoid = 0,
           float inflation = 0.0f)
      : min_agents(min_agents),
        vibes(vibes),
        input_resources(inputs),
        output_resources(outputs),
        cooldown(cooldown),
        sigmoid(sigmoid),
        inflation(inflation),
        activation_count(0) {}

  // Calculate cost multiplier based on activation count
  // - Linear phase (0 to sigmoid uses): scales from 0 (free) to 1 (full price)
  // - Exponential phase (after sigmoid uses): multiplier = (1 + inflation) ^ (activation_count - sigmoid)
  float get_cost_multiplier() const {
    // Linear phase: free to full price for first 'sigmoid' uses
    if (sigmoid > 0 && activation_count < sigmoid) {
      return static_cast<float>(activation_count) / static_cast<float>(sigmoid);
    }
    // Exponential phase: inflating costs after sigmoid uses
    if (inflation == 0.0f) return 1.0f;
    int exponent = static_cast<int>(activation_count) - static_cast<int>(sigmoid);
    return std::pow(1.0f + inflation, static_cast<float>(exponent));
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
      .def_readwrite("sigmoid", &Protocol::sigmoid)
      .def_readwrite("inflation", &Protocol::inflation)
      .def_readwrite("activation_count", &Protocol::activation_count);
}

#endif  // PACKAGES_METTAGRID_CPP_INCLUDE_METTAGRID_OBJECTS_PROTOCOL_HPP_
