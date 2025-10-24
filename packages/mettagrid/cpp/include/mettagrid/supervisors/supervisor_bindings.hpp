#ifndef PACKAGES_METTAGRID_CPP_INCLUDE_METTAGRID_SUPERVISORS_SUPERVISOR_BINDINGS_HPP_
#define PACKAGES_METTAGRID_CPP_INCLUDE_METTAGRID_SUPERVISORS_SUPERVISOR_BINDINGS_HPP_

#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include "supervisors/agent_supervisor.hpp"
#include "supervisors/resource_transport_supervisor.hpp"

namespace py = pybind11;

inline void bind_supervisor_configs(py::module& m) {
  // Base AgentSupervisorConfig
  py::class_<AgentSupervisorConfig, std::shared_ptr<AgentSupervisorConfig>>(m, "AgentSupervisorConfig")
      .def(py::init<bool, const std::string&>(), py::arg("can_override_action") = false, py::arg("name") = "supervisor")
      .def_readwrite("can_override_action", &AgentSupervisorConfig::can_override_action)
      .def_readwrite("name", &AgentSupervisorConfig::name);

  // ResourceTransportSupervisorConfig
  py::class_<ResourceTransportSupervisorConfig,
             AgentSupervisorConfig,
             std::shared_ptr<ResourceTransportSupervisorConfig>>(m, "ResourceTransportSupervisorConfig")
      .def(py::init<InventoryItem, InventoryQuantity, bool, GridCoord, bool, const std::string&>(),
           py::arg("target_resource"),
           py::arg("min_energy_threshold") = 10,
           py::arg("manage_energy") = true,
           py::arg("max_search_distance") = 30,
           py::arg("can_override_action") = false,
           py::arg("name") = "resource_transport_supervisor")
      .def_readwrite("target_resource", &ResourceTransportSupervisorConfig::target_resource)
      .def_readwrite("min_energy_threshold", &ResourceTransportSupervisorConfig::min_energy_threshold)
      .def_readwrite("manage_energy", &ResourceTransportSupervisorConfig::manage_energy)
      .def_readwrite("max_search_distance", &ResourceTransportSupervisorConfig::max_search_distance);
}

#endif  // PACKAGES_METTAGRID_CPP_INCLUDE_METTAGRID_SUPERVISORS_SUPERVISOR_BINDINGS_HPP_
