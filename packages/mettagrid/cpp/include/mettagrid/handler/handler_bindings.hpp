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

  // HandlerType enum
  py::enum_<HandlerType>(m, "HandlerType")
      .value("on_use", HandlerType::on_use)
      .value("on_update", HandlerType::on_update)
      .value("aoe", HandlerType::aoe);

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
      .def_readwrite("resource_ids", &ClearInventoryMutationConfig::resource_ids);

  py::class_<AttackMutationConfig>(m, "AttackMutationConfig")
      .def(py::init<>())
      .def_readwrite("weapon_resource", &AttackMutationConfig::weapon_resource)
      .def_readwrite("armor_resource", &AttackMutationConfig::armor_resource)
      .def_readwrite("health_resource", &AttackMutationConfig::health_resource)
      .def_readwrite("damage_multiplier", &AttackMutationConfig::damage_multiplier);

  // HandlerConfig with methods to add filters and mutations
  py::class_<HandlerConfig, std::shared_ptr<HandlerConfig>>(m, "HandlerConfig")
      .def(py::init<>())
      .def(py::init<const std::string&>(), py::arg("name"))
      .def_readwrite("name", &HandlerConfig::name)
      .def_readwrite("radius", &HandlerConfig::radius)
      // Add filter methods - each type wraps into the variant
      .def(
          "add_vibe_filter",
          [](HandlerConfig& self, const VibeFilterConfig& cfg) { self.filters.push_back(cfg); },
          py::arg("filter"))
      .def(
          "add_resource_filter",
          [](HandlerConfig& self, const ResourceFilterConfig& cfg) { self.filters.push_back(cfg); },
          py::arg("filter"))
      .def(
          "add_alignment_filter",
          [](HandlerConfig& self, const AlignmentFilterConfig& cfg) { self.filters.push_back(cfg); },
          py::arg("filter"))
      .def(
          "add_tag_filter",
          [](HandlerConfig& self, const TagFilterConfig& cfg) { self.filters.push_back(cfg); },
          py::arg("filter"))
      // Add mutation methods - each type wraps into the variant
      .def(
          "add_resource_delta_mutation",
          [](HandlerConfig& self, const ResourceDeltaMutationConfig& cfg) { self.mutations.push_back(cfg); },
          py::arg("mutation"))
      .def(
          "add_resource_transfer_mutation",
          [](HandlerConfig& self, const ResourceTransferMutationConfig& cfg) { self.mutations.push_back(cfg); },
          py::arg("mutation"))
      .def(
          "add_alignment_mutation",
          [](HandlerConfig& self, const AlignmentMutationConfig& cfg) { self.mutations.push_back(cfg); },
          py::arg("mutation"))
      .def(
          "add_freeze_mutation",
          [](HandlerConfig& self, const FreezeMutationConfig& cfg) { self.mutations.push_back(cfg); },
          py::arg("mutation"))
      .def(
          "add_clear_inventory_mutation",
          [](HandlerConfig& self, const ClearInventoryMutationConfig& cfg) { self.mutations.push_back(cfg); },
          py::arg("mutation"))
      .def(
          "add_attack_mutation",
          [](HandlerConfig& self, const AttackMutationConfig& cfg) { self.mutations.push_back(cfg); },
          py::arg("mutation"));
}

#endif  // PACKAGES_METTAGRID_CPP_INCLUDE_METTAGRID_HANDLER_HANDLER_BINDINGS_HPP_
