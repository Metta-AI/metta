#ifndef PACKAGES_METTAGRID_CPP_INCLUDE_METTAGRID_ACTIONS_ACTIVATION_HANDLER_CONFIG_HPP_
#define PACKAGES_METTAGRID_CPP_INCLUDE_METTAGRID_ACTIONS_ACTIVATION_HANDLER_CONFIG_HPP_

#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include <memory>
#include <string>
#include <unordered_map>
#include <vector>

#include "core/types.hpp"

namespace py = pybind11;

// ===== Enums for Activation Handlers =====
// These provide type-safe comparisons instead of string comparisons

// Target type for filters and mutations
enum class TargetType : uint8_t {
  Actor = 0,
  Target = 1,
  ActorCommons = 2,
  TargetCommons = 3,
};

// Filter types
enum class FilterType : uint8_t {
  Vibe = 0,
  Resource = 1,
  Alignment = 2,
};

// Mutation types
enum class MutationType : uint8_t {
  ResourceDelta = 0,
  ResourceTransfer = 1,
  Alignment = 2,
  Freeze = 3,
  Attack = 4,
  ClearInventory = 5,
};

// Alignment check types
enum class AlignmentType : uint8_t {
  Aligned = 0,
  Unaligned = 1,
  SameCommons = 2,
  DifferentCommons = 3,
  NotSameCommons = 4,
};

// Align-to mutation targets
enum class AlignToType : uint8_t {
  None = 0,
  ActorCommons = 1,
};

// ===== Config Structures for Activation Handlers =====
// These are POD types for pybind11 conversion from Python
// They don't depend on runtime types like Agent, Grid, etc.

struct VibeFilterConfig {
  TargetType target = TargetType::Actor;
  ObservationType vibe = 0;
};

struct ResourceFilterConfig {
  TargetType target = TargetType::Actor;
  std::unordered_map<InventoryItem, InventoryQuantity> resources;
};

struct AlignmentFilterConfig {
  TargetType target = TargetType::Target;
  AlignmentType alignment = AlignmentType::Aligned;
};

struct ResourceDeltaMutationConfig {
  TargetType target = TargetType::Actor;
  std::unordered_map<InventoryItem, InventoryDelta> deltas;
};

struct ResourceTransferMutationConfig {
  TargetType from_target = TargetType::Actor;
  TargetType to_target = TargetType::Target;
  std::unordered_map<InventoryItem, int> resources;
};

struct AlignmentMutationConfig {
  AlignToType align_to = AlignToType::None;
};

struct FreezeMutationConfig {
  TargetType target = TargetType::Target;
  int duration = 0;
};

struct ClearInventoryMutationConfig {
  TargetType target = TargetType::Target;
  std::string limit_name;  // Name of the resource limit group to clear (e.g., "gear")
};

// Forward declaration for recursive type
struct ActivationMutationConfig;

struct AttackMutationConfig {
  std::unordered_map<InventoryItem, InventoryQuantity> defense_resources;
  std::unordered_map<InventoryItem, InventoryQuantity> armor_resources;
  std::unordered_map<InventoryItem, InventoryQuantity> weapon_resources;
  std::unordered_map<ObservationType, int> vibe_bonus;
  std::vector<std::shared_ptr<ActivationMutationConfig>> on_success;
};

struct ActivationMutationConfig {
  MutationType type = MutationType::ResourceDelta;
  // Only one of these will be set based on type
  ResourceDeltaMutationConfig resource_delta;
  ResourceTransferMutationConfig resource_transfer;
  AlignmentMutationConfig alignment;
  FreezeMutationConfig freeze;
  AttackMutationConfig attack;
  ClearInventoryMutationConfig clear_inventory;
};

struct ActivationFilterConfig {
  FilterType type = FilterType::Vibe;
  VibeFilterConfig vibe;
  ResourceFilterConfig resource;
  AlignmentFilterConfig alignment;
};

struct ActivationHandlerConfig {
  std::string name;
  std::vector<ActivationFilterConfig> filters;
  std::vector<ActivationMutationConfig> mutations;
};

// ===== Pybind11 Bindings =====

inline void bind_activation_handler_configs(py::module& m) {
  // Bind enums first
  py::enum_<TargetType>(m, "TargetType")
      .value("Actor", TargetType::Actor)
      .value("Target", TargetType::Target)
      .value("ActorCommons", TargetType::ActorCommons)
      .value("TargetCommons", TargetType::TargetCommons);

  py::enum_<FilterType>(m, "FilterType")
      .value("Vibe", FilterType::Vibe)
      .value("Resource", FilterType::Resource)
      .value("Alignment", FilterType::Alignment);

  py::enum_<MutationType>(m, "MutationType")
      .value("ResourceDelta", MutationType::ResourceDelta)
      .value("ResourceTransfer", MutationType::ResourceTransfer)
      .value("Alignment", MutationType::Alignment)
      .value("Freeze", MutationType::Freeze)
      .value("Attack", MutationType::Attack)
      .value("ClearInventory", MutationType::ClearInventory);

  py::enum_<AlignmentType>(m, "AlignmentType")
      .value("Aligned", AlignmentType::Aligned)
      .value("Unaligned", AlignmentType::Unaligned)
      .value("SameCommons", AlignmentType::SameCommons)
      .value("DifferentCommons", AlignmentType::DifferentCommons)
      .value("NotSameCommons", AlignmentType::NotSameCommons);

  py::enum_<AlignToType>(m, "AlignToType")
      .value("None_", AlignToType::None)
      .value("ActorCommons", AlignToType::ActorCommons);

  py::class_<VibeFilterConfig>(m, "VibeFilterConfig")
      .def(py::init<>())
      .def_readwrite("target", &VibeFilterConfig::target)
      .def_readwrite("vibe", &VibeFilterConfig::vibe);

  py::class_<ResourceFilterConfig>(m, "ResourceFilterConfig")
      .def(py::init<>())
      .def_readwrite("target", &ResourceFilterConfig::target)
      .def_readwrite("resources", &ResourceFilterConfig::resources);

  py::class_<AlignmentFilterConfig>(m, "AlignmentFilterConfig")
      .def(py::init<>())
      .def_readwrite("target", &AlignmentFilterConfig::target)
      .def_readwrite("alignment", &AlignmentFilterConfig::alignment);

  py::class_<ResourceDeltaMutationConfig>(m, "ResourceDeltaMutationConfig")
      .def(py::init<>())
      .def_readwrite("target", &ResourceDeltaMutationConfig::target)
      .def_readwrite("deltas", &ResourceDeltaMutationConfig::deltas);

  py::class_<ResourceTransferMutationConfig>(m, "ResourceTransferMutationConfig")
      .def(py::init<>())
      .def_readwrite("from_target", &ResourceTransferMutationConfig::from_target)
      .def_readwrite("to_target", &ResourceTransferMutationConfig::to_target)
      .def_readwrite("resources", &ResourceTransferMutationConfig::resources);

  py::class_<AlignmentMutationConfig>(m, "AlignmentMutationConfig")
      .def(py::init<>())
      .def_readwrite("align_to", &AlignmentMutationConfig::align_to);

  py::class_<FreezeMutationConfig>(m, "FreezeMutationConfig")
      .def(py::init<>())
      .def_readwrite("target", &FreezeMutationConfig::target)
      .def_readwrite("duration", &FreezeMutationConfig::duration);

  py::class_<ClearInventoryMutationConfig>(m, "ClearInventoryMutationConfig")
      .def(py::init<>())
      .def_readwrite("target", &ClearInventoryMutationConfig::target)
      .def_readwrite("limit_name", &ClearInventoryMutationConfig::limit_name);

  py::class_<AttackMutationConfig>(m, "AttackMutationConfig")
      .def(py::init<>())
      .def_readwrite("defense_resources", &AttackMutationConfig::defense_resources)
      .def_readwrite("armor_resources", &AttackMutationConfig::armor_resources)
      .def_readwrite("weapon_resources", &AttackMutationConfig::weapon_resources)
      .def_readwrite("vibe_bonus", &AttackMutationConfig::vibe_bonus)
      .def_readwrite("on_success", &AttackMutationConfig::on_success);

  py::class_<ActivationMutationConfig, std::shared_ptr<ActivationMutationConfig>>(m, "ActivationMutationConfig")
      .def(py::init<>())
      .def_readwrite("type", &ActivationMutationConfig::type)
      .def_readwrite("resource_delta", &ActivationMutationConfig::resource_delta)
      .def_readwrite("resource_transfer", &ActivationMutationConfig::resource_transfer)
      .def_readwrite("alignment", &ActivationMutationConfig::alignment)
      .def_readwrite("freeze", &ActivationMutationConfig::freeze)
      .def_readwrite("attack", &ActivationMutationConfig::attack)
      .def_readwrite("clear_inventory", &ActivationMutationConfig::clear_inventory);

  py::class_<ActivationFilterConfig>(m, "ActivationFilterConfig")
      .def(py::init<>())
      .def_readwrite("type", &ActivationFilterConfig::type)
      .def_readwrite("vibe", &ActivationFilterConfig::vibe)
      .def_readwrite("resource", &ActivationFilterConfig::resource)
      .def_readwrite("alignment", &ActivationFilterConfig::alignment);

  py::class_<ActivationHandlerConfig>(m, "ActivationHandlerConfig")
      .def(py::init<>())
      .def_readwrite("name", &ActivationHandlerConfig::name)
      .def_readwrite("filters", &ActivationHandlerConfig::filters)
      .def_readwrite("mutations", &ActivationHandlerConfig::mutations);
}

#endif  // PACKAGES_METTAGRID_CPP_INCLUDE_METTAGRID_ACTIONS_ACTIVATION_HANDLER_CONFIG_HPP_
