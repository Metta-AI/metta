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

// ===== Config Structures for Activation Handlers =====
// These are POD types for pybind11 conversion from Python
// They don't depend on runtime types like Agent, Grid, etc.

struct VibeFilterConfig {
  std::string target = "actor";  // "actor" or "target"
  ObservationType vibe = 0;
};

struct ResourceFilterConfig {
  std::string target = "actor";
  std::unordered_map<InventoryItem, InventoryQuantity> resources;
};

struct ResourceDeltaMutationConfig {
  std::string target;
  std::unordered_map<InventoryItem, InventoryDelta> deltas;
};

struct ResourceTransferMutationConfig {
  std::string from_target;
  std::string to_target;
  std::unordered_map<InventoryItem, int> resources;
};

struct AlignmentMutationConfig {
  std::string align_to;
};

struct FreezeMutationConfig {
  std::string target;
  int duration = 0;
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
  std::string type;  // "resource_delta", "resource_transfer", "alignment", "freeze", "attack"
  // Only one of these will be set based on type
  ResourceDeltaMutationConfig resource_delta;
  ResourceTransferMutationConfig resource_transfer;
  AlignmentMutationConfig alignment;
  FreezeMutationConfig freeze;
  AttackMutationConfig attack;
};

struct ActivationFilterConfig {
  std::string type;  // "vibe", "resource"
  VibeFilterConfig vibe;
  ResourceFilterConfig resource;
};

struct ActivationHandlerConfig {
  std::string name;
  std::vector<ActivationFilterConfig> filters;
  std::vector<ActivationMutationConfig> mutations;
};

// ===== Pybind11 Bindings =====

inline void bind_activation_handler_configs(py::module& m) {
  py::class_<VibeFilterConfig>(m, "VibeFilterConfig")
      .def(py::init<>())
      .def_readwrite("target", &VibeFilterConfig::target)
      .def_readwrite("vibe", &VibeFilterConfig::vibe);

  py::class_<ResourceFilterConfig>(m, "ResourceFilterConfig")
      .def(py::init<>())
      .def_readwrite("target", &ResourceFilterConfig::target)
      .def_readwrite("resources", &ResourceFilterConfig::resources);

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
      .def_readwrite("attack", &ActivationMutationConfig::attack);

  py::class_<ActivationFilterConfig>(m, "ActivationFilterConfig")
      .def(py::init<>())
      .def_readwrite("type", &ActivationFilterConfig::type)
      .def_readwrite("vibe", &ActivationFilterConfig::vibe)
      .def_readwrite("resource", &ActivationFilterConfig::resource);

  py::class_<ActivationHandlerConfig>(m, "ActivationHandlerConfig")
      .def(py::init<>())
      .def_readwrite("name", &ActivationHandlerConfig::name)
      .def_readwrite("filters", &ActivationHandlerConfig::filters)
      .def_readwrite("mutations", &ActivationHandlerConfig::mutations);
}

#endif  // PACKAGES_METTAGRID_CPP_INCLUDE_METTAGRID_ACTIONS_ACTIVATION_HANDLER_CONFIG_HPP_

