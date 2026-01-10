#ifndef PACKAGES_METTAGRID_CPP_INCLUDE_METTAGRID_CORE_AOE_BINDINGS_HPP_
#define PACKAGES_METTAGRID_CPP_INCLUDE_METTAGRID_CORE_AOE_BINDINGS_HPP_

#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include "core/aoe_config.hpp"

namespace py = pybind11;

inline void bind_aoe_config(py::module& m) {
  using namespace mettagrid;

  // AOEAlignmentFilter enum
  py::enum_<AOEAlignmentFilter>(m, "AOEAlignmentFilter")
      .value("any", AOEAlignmentFilter::any)
      .value("same_collective", AOEAlignmentFilter::same_collective)
      .value("different_collective", AOEAlignmentFilter::different_collective);

  // AOEResourceDelta struct
  py::class_<AOEResourceDelta>(m, "AOEResourceDelta")
      .def(py::init<>())
      .def(py::init<InventoryItem, InventoryDelta>(), py::arg("resource_id"), py::arg("delta"))
      .def_readwrite("resource_id", &AOEResourceDelta::resource_id)
      .def_readwrite("delta", &AOEResourceDelta::delta);

  // AOEConfig struct
  py::class_<AOEConfig, std::shared_ptr<AOEConfig>>(m, "AOEConfig")
      .def(py::init<>())
      .def_readwrite("radius", &AOEConfig::radius)
      .def_readwrite("deltas", &AOEConfig::deltas)
      .def_readwrite("target_tag_ids", &AOEConfig::target_tag_ids)
      .def_readwrite("alignment_filter", &AOEConfig::alignment_filter);
}

#endif  // PACKAGES_METTAGRID_CPP_INCLUDE_METTAGRID_CORE_AOE_BINDINGS_HPP_
