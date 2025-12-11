// agent_config.hpp
#ifndef PACKAGES_METTAGRID_CPP_INCLUDE_METTAGRID_OBJECTS_AGENT_CONFIG_HPP_
#define PACKAGES_METTAGRID_CPP_INCLUDE_METTAGRID_OBJECTS_AGENT_CONFIG_HPP_

#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include <memory>
#include <string>
#include <unordered_map>

#include "core/grid_object.hpp"
#include "core/types.hpp"
#include "objects/inventory_config.hpp"

// Configuration for damage system: when all threshold inventory resources are reached,
// one random resource from the resources map is destroyed (weighted by quantity above minimum)
struct DamageConfig {
  // Map of inventory item to threshold values. All must be reached to trigger damage.
  std::unordered_map<InventoryItem, int> threshold;
  // Map of inventory items that can be destroyed, with their minimum values.
  // Only resources listed here can be destroyed. Resources at or below minimum are protected.
  std::unordered_map<InventoryItem, int> resources;

  bool enabled() const {
    return !threshold.empty() && !resources.empty();
  }
};

struct AgentConfig : public GridObjectConfig {
  AgentConfig(TypeId type_id,
              const std::string& type_name,
              unsigned char group_id,
              const std::string& group_name,
              unsigned char freeze_duration = 0,
              const InventoryConfig& inventory_config = InventoryConfig(),
              const std::unordered_map<std::string, RewardType>& stat_rewards = {},
              const std::unordered_map<std::string, RewardType>& stat_reward_max = {},
              const std::unordered_map<InventoryItem, InventoryQuantity>& initial_inventory = {},
              const std::vector<InventoryItem>& soul_bound_resources = {},
              const std::unordered_map<InventoryItem, InventoryQuantity>& inventory_regen_amounts = {},
              const std::vector<InventoryItem>& diversity_tracked_resources = {},
              const std::unordered_map<ObservationType, std::unordered_map<InventoryItem, int>>& vibe_transfers = {},
              ObservationType initial_vibe = 0,
              const DamageConfig& damage_config = DamageConfig())
      : GridObjectConfig(type_id, type_name, initial_vibe),
        group_id(group_id),
        group_name(group_name),
        freeze_duration(freeze_duration),
        inventory_config(inventory_config),
        stat_rewards(stat_rewards),
        stat_reward_max(stat_reward_max),
        initial_inventory(initial_inventory),
        soul_bound_resources(soul_bound_resources),
        inventory_regen_amounts(inventory_regen_amounts),
        diversity_tracked_resources(diversity_tracked_resources),
        vibe_transfers(vibe_transfers),
        damage_config(damage_config) {}

  unsigned char group_id;
  std::string group_name;
  short freeze_duration;
  InventoryConfig inventory_config;
  std::unordered_map<std::string, RewardType> stat_rewards;
  std::unordered_map<std::string, RewardType> stat_reward_max;
  std::unordered_map<InventoryItem, InventoryQuantity> initial_inventory;
  std::vector<InventoryItem> soul_bound_resources;
  std::unordered_map<InventoryItem, InventoryQuantity> inventory_regen_amounts;
  std::vector<InventoryItem> diversity_tracked_resources;
  // Maps vibe to resource deltas for agent-to-agent sharing
  std::unordered_map<ObservationType, std::unordered_map<InventoryItem, int>> vibe_transfers;
  DamageConfig damage_config;
};

namespace py = pybind11;

inline void bind_damage_config(py::module& m) {
  py::class_<DamageConfig>(m, "DamageConfig")
      .def(py::init<>())
      .def_readwrite("threshold", &DamageConfig::threshold)
      .def_readwrite("resources", &DamageConfig::resources)
      .def("enabled", &DamageConfig::enabled);
}

inline void bind_agent_config(py::module& m) {
  bind_damage_config(m);

  py::class_<AgentConfig, GridObjectConfig, std::shared_ptr<AgentConfig>>(m, "AgentConfig")
      .def(py::init<TypeId,
                    const std::string&,
                    unsigned char,
                    const std::string&,
                    unsigned char,
                    const InventoryConfig&,
                    const std::unordered_map<std::string, RewardType>&,
                    const std::unordered_map<std::string, RewardType>&,
                    const std::unordered_map<InventoryItem, InventoryQuantity>&,
                    const std::vector<InventoryItem>&,
                    const std::unordered_map<InventoryItem, InventoryQuantity>&,
                    const std::vector<InventoryItem>&,
                    const std::unordered_map<ObservationType, std::unordered_map<InventoryItem, int>>&,
                    ObservationType,
                    const DamageConfig&>(),
           py::arg("type_id"),
           py::arg("type_name") = "agent",
           py::arg("group_id"),
           py::arg("group_name"),
           py::arg("freeze_duration") = 0,
           py::arg("inventory_config") = InventoryConfig(),
           py::arg("stat_rewards") = std::unordered_map<std::string, RewardType>(),
           py::arg("stat_reward_max") = std::unordered_map<std::string, RewardType>(),
           py::arg("initial_inventory") = std::unordered_map<InventoryItem, InventoryQuantity>(),
           py::arg("soul_bound_resources") = std::vector<InventoryItem>(),
           py::arg("inventory_regen_amounts") = std::unordered_map<InventoryItem, InventoryQuantity>(),
           py::arg("diversity_tracked_resources") = std::vector<InventoryItem>(),
           py::arg("vibe_transfers") = std::unordered_map<ObservationType, std::unordered_map<InventoryItem, int>>(),
           py::arg("initial_vibe") = 0,
           py::arg("damage_config") = DamageConfig())
      .def_readwrite("type_id", &AgentConfig::type_id)
      .def_readwrite("type_name", &AgentConfig::type_name)
      .def_readwrite("tag_ids", &AgentConfig::tag_ids)
      .def_readwrite("group_name", &AgentConfig::group_name)
      .def_readwrite("group_id", &AgentConfig::group_id)
      .def_readwrite("freeze_duration", &AgentConfig::freeze_duration)
      .def_readwrite("inventory_config", &AgentConfig::inventory_config)
      .def_readwrite("stat_rewards", &AgentConfig::stat_rewards)
      .def_readwrite("stat_reward_max", &AgentConfig::stat_reward_max)
      .def_readwrite("initial_inventory", &AgentConfig::initial_inventory)
      .def_readwrite("soul_bound_resources", &AgentConfig::soul_bound_resources)
      .def_readwrite("inventory_regen_amounts", &AgentConfig::inventory_regen_amounts)
      .def_readwrite("diversity_tracked_resources", &AgentConfig::diversity_tracked_resources)
      .def_readwrite("vibe_transfers", &AgentConfig::vibe_transfers)
      .def_readwrite("initial_vibe", &AgentConfig::initial_vibe)
      .def_readwrite("damage_config", &AgentConfig::damage_config);
}

#endif  // PACKAGES_METTAGRID_CPP_INCLUDE_METTAGRID_OBJECTS_AGENT_CONFIG_HPP_
