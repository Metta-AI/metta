#ifndef PACKAGES_METTAGRID_CPP_INCLUDE_METTAGRID_SUPERVISORS_SUPERVISOR_BINDINGS_HPP_
#define PACKAGES_METTAGRID_CPP_INCLUDE_METTAGRID_SUPERVISORS_SUPERVISOR_BINDINGS_HPP_

#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include "supervisors/agent_supervisor.hpp"
#include "supervisors/patrol_supervisor.hpp"

namespace py = pybind11;

inline void bind_supervisor_configs(py::module& m) {
  // Base AgentSupervisorConfig
  py::class_<AgentSupervisorConfig, std::shared_ptr<AgentSupervisorConfig>>(m, "AgentSupervisorConfig")
      .def(py::init<bool, const std::string&>(), py::arg("can_override_action") = false, py::arg("name") = "supervisor")
      .def_readwrite("can_override_action", &AgentSupervisorConfig::can_override_action)
      .def_readwrite("name", &AgentSupervisorConfig::name);

  // PatrolSupervisorConfig
  py::class_<PatrolSupervisorConfig, AgentSupervisorConfig, std::shared_ptr<PatrolSupervisorConfig>>(
      m, "PatrolSupervisorConfig")
      .def(py::init<int, bool, const std::string&>(),
           py::arg("steps_per_direction") = 5,
           py::arg("can_override_action") = false,
           py::arg("name") = "patrol_supervisor")
      .def_readwrite("steps_per_direction", &PatrolSupervisorConfig::steps_per_direction);
}

#endif  // PACKAGES_METTAGRID_CPP_INCLUDE_METTAGRID_SUPERVISORS_SUPERVISOR_BINDINGS_HPP_
