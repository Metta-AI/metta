// agent_config.hpp
#ifndef PACKAGES_METTAGRID_CPP_INCLUDE_METTAGRID_OBJECTS_AGENT_CONFIG_HPP_
#define PACKAGES_METTAGRID_CPP_INCLUDE_METTAGRID_OBJECTS_AGENT_CONFIG_HPP_

#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include <map>
#include <string>

#include "core/grid_object.hpp"
#include "core/types.hpp"
#include "objects/inventory_config.hpp"

struct AgentConfig : public GridObjectConfig {
  AgentConfig(TypeId type_id,
              const std::string& type_name,
              unsigned char group_id,
              const std::string& group_name,
              unsigned char freeze_duration = 0,
              float action_failure_penalty = 0,
              const InventoryConfig& inventory_config = InventoryConfig(),
              const std::map<std::string, RewardType>& stat_rewards = {},
              const std::map<std::string, RewardType>& stat_reward_max = {},
              float group_reward_pct = 0,
              const std::map<InventoryItem, InventoryQuantity>& initial_inventory = {},
              const std::vector<int>& tag_ids = {},
              const std::vector<InventoryItem>& soul_bound_resources = {})
      : GridObjectConfig(type_id, type_name, tag_ids),
        group_id(group_id),
        group_name(group_name),
        freeze_duration(freeze_duration),
        action_failure_penalty(action_failure_penalty),
        inventory_config(inventory_config),
        stat_rewards(stat_rewards),
        stat_reward_max(stat_reward_max),
        group_reward_pct(group_reward_pct),
        initial_inventory(initial_inventory),
        soul_bound_resources(soul_bound_resources) {}

  unsigned char group_id;
  std::string group_name;
  short freeze_duration;
  float action_failure_penalty;
  InventoryConfig inventory_config;
  std::map<std::string, RewardType> stat_rewards;
  std::map<std::string, RewardType> stat_reward_max;
  float group_reward_pct;
  std::map<InventoryItem, InventoryQuantity> initial_inventory;
  std::vector<InventoryItem> soul_bound_resources;
};

namespace py = pybind11;

inline void bind_agent_config(py::module& m) {
  py::class_<AgentConfig, GridObjectConfig, std::shared_ptr<AgentConfig>>(m, "AgentConfig")
      .def(py::init<TypeId,
                    const std::string&,
                    unsigned char,
                    const std::string&,
                    unsigned char,
                    float,
                    const InventoryConfig&,
                    const std::map<std::string, RewardType>&,
                    const std::map<std::string, RewardType>&,
                    float,
                    const std::map<InventoryItem, InventoryQuantity>&,
                    const std::vector<int>&,
                    const std::vector<InventoryItem>&>(),
           py::arg("type_id"),
           py::arg("type_name") = "agent",
           py::arg("group_id"),
           py::arg("group_name"),
           py::arg("freeze_duration") = 0,
           py::arg("action_failure_penalty") = 0,
           py::arg("inventory_config") = InventoryConfig(),
           py::arg("stat_rewards") = std::map<std::string, RewardType>(),
           py::arg("stat_reward_max") = std::map<std::string, RewardType>(),
           py::arg("group_reward_pct") = 0,
           py::arg("initial_inventory") = std::map<InventoryItem, InventoryQuantity>(),
           py::arg("tag_ids") = std::vector<int>(),
           py::arg("soul_bound_resources") = std::vector<InventoryItem>())
      .def_readwrite("type_id", &GridObjectConfig::type_id)
      .def_readwrite("type_name", &GridObjectConfig::type_name)
      .def_readwrite("group_name", &AgentConfig::group_name)
      .def_readwrite("group_id", &AgentConfig::group_id)
      .def_readwrite("freeze_duration", &AgentConfig::freeze_duration)
      .def_readwrite("action_failure_penalty", &AgentConfig::action_failure_penalty)
      .def_readwrite("inventory_config", &AgentConfig::inventory_config)
      .def_readwrite("stat_rewards", &AgentConfig::stat_rewards)
      .def_readwrite("stat_reward_max", &AgentConfig::stat_reward_max)
      .def_readwrite("group_reward_pct", &AgentConfig::group_reward_pct)
      .def_readwrite("initial_inventory", &AgentConfig::initial_inventory)
      .def_readwrite("tag_ids", &GridObjectConfig::tag_ids)
      .def_readwrite("soul_bound_resources", &AgentConfig::soul_bound_resources);
}

#endif  // PACKAGES_METTAGRID_CPP_INCLUDE_METTAGRID_OBJECTS_AGENT_CONFIG_HPP_
