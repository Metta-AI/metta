#ifndef PACKAGES_METTAGRID_CPP_INCLUDE_METTAGRID_HANDLER_HANDLER_BINDINGS_HPP_
#define PACKAGES_METTAGRID_CPP_INCLUDE_METTAGRID_HANDLER_HANDLER_BINDINGS_HPP_

#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include "handler/handler_config.hpp"

namespace py = pybind11;

inline void bind_handler_config(py::module& m) {
  using namespace mettagrid;

  // EntityRef enum
  py::enum_<EntityRef>(m, "EntityRef")
      .value("actor", EntityRef::actor)
      .value("target", EntityRef::target)
      .value("actor_collective", EntityRef::actor_collective)
      .value("target_collective", EntityRef::target_collective);

  // AlignmentCondition enum
  py::enum_<AlignmentCondition>(m, "AlignmentCondition")
      .value("aligned", AlignmentCondition::aligned)
      .value("unaligned", AlignmentCondition::unaligned)
      .value("same_collective", AlignmentCondition::same_collective)
      .value("different_collective", AlignmentCondition::different_collective);

  // AlignTo enum
  py::enum_<AlignTo>(m, "AlignTo").value("actor_collective", AlignTo::actor_collective).value("none", AlignTo::none);

  // Filter configs
  py::class_<VibeFilterConfig>(m, "VibeFilterConfig")
      .def(py::init<>())
      .def_readwrite("entity", &VibeFilterConfig::entity)
      .def_readwrite("vibe_id", &VibeFilterConfig::vibe_id);

  py::class_<ResourceFilterConfig>(m, "ResourceFilterConfig")
      .def(py::init<>())
      .def_readwrite("entity", &ResourceFilterConfig::entity)
      .def_readwrite("resource_id", &ResourceFilterConfig::resource_id)
      .def_readwrite("min_amount", &ResourceFilterConfig::min_amount);

  py::class_<AlignmentFilterConfig>(m, "AlignmentFilterConfig")
      .def(py::init<>())
      .def_readwrite("condition", &AlignmentFilterConfig::condition);

  py::class_<TagFilterConfig>(m, "TagFilterConfig")
      .def(py::init<>())
      .def_readwrite("entity", &TagFilterConfig::entity)
      .def_readwrite("required_tag_ids", &TagFilterConfig::required_tag_ids);

  // Mutation configs
  py::class_<ResourceDeltaMutationConfig>(m, "ResourceDeltaMutationConfig")
      .def(py::init<>())
      .def_readwrite("entity", &ResourceDeltaMutationConfig::entity)
      .def_readwrite("resource_id", &ResourceDeltaMutationConfig::resource_id)
      .def_readwrite("delta", &ResourceDeltaMutationConfig::delta);

  py::class_<ResourceTransferMutationConfig>(m, "ResourceTransferMutationConfig")
      .def(py::init<>())
      .def_readwrite("source", &ResourceTransferMutationConfig::source)
      .def_readwrite("destination", &ResourceTransferMutationConfig::destination)
      .def_readwrite("resource_id", &ResourceTransferMutationConfig::resource_id)
      .def_readwrite("amount", &ResourceTransferMutationConfig::amount);

  py::class_<AlignmentMutationConfig>(m, "AlignmentMutationConfig")
      .def(py::init<>())
      .def_readwrite("align_to", &AlignmentMutationConfig::align_to);

  py::class_<FreezeMutationConfig>(m, "FreezeMutationConfig")
      .def(py::init<>())
      .def_readwrite("duration", &FreezeMutationConfig::duration);

  py::class_<ClearInventoryMutationConfig>(m, "ClearInventoryMutationConfig")
      .def(py::init<>())
      .def_readwrite("entity", &ClearInventoryMutationConfig::entity)
      .def_readwrite("resource_id", &ClearInventoryMutationConfig::resource_id);

  py::class_<AttackMutationConfig>(m, "AttackMutationConfig")
      .def(py::init<>())
      .def_readwrite("weapon_resource", &AttackMutationConfig::weapon_resource)
      .def_readwrite("armor_resource", &AttackMutationConfig::armor_resource)
      .def_readwrite("health_resource", &AttackMutationConfig::health_resource)
      .def_readwrite("damage_multiplier", &AttackMutationConfig::damage_multiplier);

  // Note: FilterConfig and MutationConfig are std::variant types.
  // For full Python support, you'd need to bind each variant alternative
  // and use py::implicitly_convertible or custom type casters.
  // For now, the individual config types above can be used directly.

  // HandlerConfig
  py::class_<HandlerConfig, std::shared_ptr<HandlerConfig>>(m, "HandlerConfig")
      .def(py::init<>())
      .def(py::init<const std::string&>(), py::arg("name"))
      .def_readwrite("name", &HandlerConfig::name);
  // Note: filters and mutations vectors contain std::variant types.
  // For full Python support, custom conversion would be needed.
}

#endif  // PACKAGES_METTAGRID_CPP_INCLUDE_METTAGRID_HANDLER_HANDLER_BINDINGS_HPP_
