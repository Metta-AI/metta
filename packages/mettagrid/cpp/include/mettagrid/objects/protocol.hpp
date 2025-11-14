#ifndef PACKAGES_METTAGRID_CPP_INCLUDE_METTAGRID_OBJECTS_PROTOCOL_HPP_
#define PACKAGES_METTAGRID_CPP_INCLUDE_METTAGRID_OBJECTS_PROTOCOL_HPP_

#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

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

  Protocol(unsigned short min_agents = 0,
           const std::vector<ObservationType>& vibes = {},
           const std::unordered_map<InventoryItem, InventoryQuantity>& inputs = {},
           const std::unordered_map<InventoryItem, InventoryQuantity>& outputs = {},
           unsigned short cooldown = 0)
      : min_agents(min_agents), vibes(vibes), input_resources(inputs), output_resources(outputs), cooldown(cooldown) {}
};

inline void bind_protocol(py::module& m) {
  py::class_<Protocol, std::shared_ptr<Protocol>>(m, "Protocol")
      .def(py::init())
      .def_readwrite("min_agents", &Protocol::min_agents)
      .def_readwrite("vibes", &Protocol::vibes)
      .def_readwrite("input_resources", &Protocol::input_resources)
      .def_readwrite("output_resources", &Protocol::output_resources)
      .def_readwrite("cooldown", &Protocol::cooldown);
}

#endif  // PACKAGES_METTAGRID_CPP_INCLUDE_METTAGRID_OBJECTS_PROTOCOL_HPP_
