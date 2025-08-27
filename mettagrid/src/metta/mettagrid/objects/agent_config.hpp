// agent_config.hpp
#ifndef OBJECTS_AGENT_CONFIG_HPP_
#define OBJECTS_AGENT_CONFIG_HPP_

#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include <map>
#include <string>

#include "../grid_object.hpp"
#include "../types.hpp"

struct AgentConfig : public GridObjectConfig {
  AgentConfig(TypeId type_id,
              const std::string& type_name,
              unsigned char group_id,
              const std::string& group_name,
              unsigned char freeze_duration = 0,
              float action_failure_penalty = 0,
              const std::map<InventoryItem, InventoryQuantity>& resource_limits = {},
              const std::map<InventoryItem, RewardType>& resource_rewards = {},
              const std::map<InventoryItem, RewardType>& resource_reward_max = {},
              const std::map<std::string, RewardType>& stat_rewards = {},
              const std::map<std::string, RewardType>& stat_reward_max = {},
              float group_reward_pct = 0,
              const std::map<InventoryItem, InventoryQuantity>& initial_inventory = {})
      : GridObjectConfig(type_id, type_name),
        group_id(group_id),
        group_name(group_name),
        freeze_duration(freeze_duration),
        action_failure_penalty(action_failure_penalty),
        resource_limits(resource_limits),
        resource_rewards(resource_rewards),
        resource_reward_max(resource_reward_max),
        stat_rewards(stat_rewards),
        stat_reward_max(stat_reward_max),
        group_reward_pct(group_reward_pct),
        initial_inventory(initial_inventory) {}

  unsigned char group_id;
  std::string group_name;
  short freeze_duration;
  float action_failure_penalty;
  std::map<InventoryItem, InventoryQuantity> resource_limits;
  std::map<InventoryItem, RewardType> resource_rewards;
  std::map<InventoryItem, RewardType> resource_reward_max;
  std::map<std::string, RewardType> stat_rewards;
  std::map<std::string, RewardType> stat_reward_max;
  float group_reward_pct;
  std::map<InventoryItem, InventoryQuantity> initial_inventory;
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
                    const std::map<InventoryItem, InventoryQuantity>&,
                    const std::map<InventoryItem, RewardType>&,
                    const std::map<InventoryItem, RewardType>&,
                    const std::map<std::string, RewardType>&,
                    const std::map<std::string, RewardType>&,
                    float,
                    const std::map<InventoryItem, InventoryQuantity>&>(),
           py::arg("type_id"),
           py::arg("type_name") = "agent",
           py::arg("group_id"),
           py::arg("group_name"),
           py::arg("freeze_duration") = 0,
           py::arg("action_failure_penalty") = 0,
           py::arg("resource_limits") = std::map<InventoryItem, InventoryQuantity>(),
           py::arg("resource_rewards") = std::map<InventoryItem, RewardType>(),
           py::arg("resource_reward_max") = std::map<InventoryItem, RewardType>(),
           py::arg("stat_rewards") = std::map<std::string, RewardType>(),
           py::arg("stat_reward_max") = std::map<std::string, RewardType>(),
           py::arg("group_reward_pct") = 0,
           py::arg("initial_inventory") = std::map<InventoryItem, InventoryQuantity>())
      .def_readwrite("type_id", &AgentConfig::type_id)
      .def_readwrite("type_name", &AgentConfig::type_name)
      .def_readwrite("group_name", &AgentConfig::group_name)
      .def_readwrite("group_id", &AgentConfig::group_id)
      .def_readwrite("freeze_duration", &AgentConfig::freeze_duration)
      .def_readwrite("action_failure_penalty", &AgentConfig::action_failure_penalty)
      .def_readwrite("resource_limits", &AgentConfig::resource_limits)
      .def_readwrite("resource_rewards", &AgentConfig::resource_rewards)
      .def_readwrite("resource_reward_max", &AgentConfig::resource_reward_max)
      .def_readwrite("stat_rewards", &AgentConfig::stat_rewards)
      .def_readwrite("stat_reward_max", &AgentConfig::stat_reward_max)
      .def_readwrite("group_reward_pct", &AgentConfig::group_reward_pct)
      .def_readwrite("initial_inventory", &AgentConfig::initial_inventory);
}

#endif  // OBJECTS_AGENT_CONFIG_HPP_
