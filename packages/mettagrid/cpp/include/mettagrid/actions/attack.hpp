#ifndef PACKAGES_METTAGRID_CPP_INCLUDE_METTAGRID_ACTIONS_ATTACK_HPP_
#define PACKAGES_METTAGRID_CPP_INCLUDE_METTAGRID_ACTIONS_ATTACK_HPP_

#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include <algorithm>
#include <map>
#include <string>
#include <vector>

#include "actions/action_handler.hpp"
#include "config/mettagrid_config.hpp"
#include "core/grid_object.hpp"
#include "core/types.hpp"
#include "objects/agent.hpp"
#include "objects/constants.hpp"

struct AttackActionConfig : public ActionConfig {
  std::map<InventoryItem, InventoryQuantity> defense_resources;

  AttackActionConfig(const std::map<InventoryItem, InventoryQuantity>& required_resources,
                     const std::map<InventoryItem, InventoryProbability>& consumed_resources,
                     const std::map<InventoryItem, InventoryQuantity>& defense_resources)
      : ActionConfig(required_resources, consumed_resources), defense_resources(defense_resources) {}
};

class Attack : public ActionHandler {
public:
  Attack(const AttackActionConfig& cfg,
         const GameConfig* game_config,
         unsigned char target_slot,
         const std::string& name)
      : ActionHandler(cfg, name),
        _defense_resources(cfg.defense_resources),
        _game_config(game_config),
        _target_slot(target_slot) {
    priority = 1;
  }

protected:
  std::map<InventoryItem, InventoryQuantity> _defense_resources;
  const GameConfig* _game_config;
  unsigned char _target_slot;

  bool _handle_action(Agent& actor) override {
    Agent* last_agent = nullptr;
    short num_skipped = 0;

    if (!_game_config->allow_diagonals) {
      static constexpr short COL_OFFSETS[3] = {0, -1, 1};

      for (short distance = 1; distance <= 3; distance++) {
        for (short offset : COL_OFFSETS) {
          GridLocation target_loc = grid().relative_location(actor.location, actor.orientation, distance, offset);
          target_loc.layer = GridLayer::AgentLayer;

          Agent* target_agent = static_cast<Agent*>(grid().object_at(target_loc));
          if (target_agent) {
            last_agent = target_agent;
            if (num_skipped == _target_slot) {
              return _handle_target(actor, *target_agent);
            }
            num_skipped++;
          }
        }
      }
    } else {
      static constexpr struct {
        short forward;
        short lateral;
      } DIAGONAL_POSITIONS[9] = {
          {1, 0},  {2, -1}, {1, 1}, {2, 0}, {3, -2},
          {1, 2}, {3, -1}, {2, 1}, {3, 0}};

      for (const auto& pos : DIAGONAL_POSITIONS) {
        GridLocation target_loc = grid().relative_location(actor.location, actor.orientation, pos.forward, pos.lateral);
        target_loc.layer = GridLayer::AgentLayer;

        Agent* target_agent = static_cast<Agent*>(grid().object_at(target_loc));
        if (target_agent) {
          last_agent = target_agent;
          if (num_skipped == _target_slot) {
            return _handle_target(actor, *target_agent);
          }
          num_skipped++;
        }
      }
    }

    if (last_agent) {
      return _handle_target(actor, *last_agent);
    }

    return false;
  }

  bool _handle_target(Agent& actor, Agent& target) {
    bool was_already_frozen = target.frozen > 0;

    if (!_defense_resources.empty()) {
      bool target_can_defend = _check_defense_capability(target);

      if (target_can_defend) {
        _consume_defense_resources(target);
        _log_blocked_attack(actor, target);
        return true;
      }
    }

    target.frozen = target.freeze_duration;

    if (!was_already_frozen) {
      _steal_resources(actor, target);
      _log_successful_attack(actor, target);
    } else {
      const std::string& actor_group = actor.group_name;
      actor.stats.incr(_action_prefix(actor_group) + "wasted_on_frozen");
    }
    return true;
  }

private:
  bool _check_defense_capability(const Agent& target) const {
    for (const auto& [item, amount] : _defense_resources) {
      if (target.inventory.amount(item) < amount) {
        return false;
      }
    }
    return true;
  }

  void _consume_defense_resources(Agent& target) {
    for (const auto& [item, amount] : _defense_resources) {
      [[maybe_unused]] InventoryDelta delta = target.update_inventory(item, -amount);
      assert(delta == -amount);
    }
  }

  void _steal_resources(Agent& actor, Agent& target) {
    std::vector<std::pair<InventoryItem, InventoryQuantity>> snapshot;
    snapshot.reserve(target.inventory.get().size());
    for (const auto& [item, amount] : target.inventory.get()) {
      snapshot.emplace_back(item, amount);
    }

    for (const auto& [item, amount] : snapshot) {
      if (std::find(target.soul_bound_resources.begin(), target.soul_bound_resources.end(), item) !=
          target.soul_bound_resources.end()) {
        continue;
      }

      InventoryDelta stolen = actor.update_inventory(item, amount);
      target.update_inventory(item, -stolen);

      if (stolen > 0) {
        _log_resource_theft(actor, target, item, stolen);
      }
    }
  }

  std::string _action_prefix(const std::string& group) const {
    return "action." + action_name() + "." + group + ".";
  }

  void _log_blocked_attack(Agent& actor, const Agent& target) const {
    const std::string& actor_group = actor.group_name;
    actor.stats.incr(_action_prefix(actor_group) + "blocked");
    actor.stats.incr(_action_prefix(actor_group) + "blocked.by_group." + target.group_name);
  }

  void _log_successful_attack(Agent& actor, const Agent& target) const {
    const std::string& actor_group = actor.group_name;
    actor.stats.incr(_action_prefix(actor_group) + "success");
    actor.stats.incr(_action_prefix(actor_group) + "success.on_group." + target.group_name);
  }

  void _log_resource_theft(Agent& actor, const Agent& target, InventoryItem item, InventoryDelta stolen) const {
    const std::string& actor_group = actor.group_name;
    actor.stats.add(_action_prefix(actor_group) + "resources." + std::to_string(item) + ".stolen",
                    static_cast<float>(stolen));
    actor.stats.add(_action_prefix(actor_group) + "resources.total_stolen", static_cast<float>(stolen));
  }
};

namespace py = pybind11;

inline void bind_attack_action_config(py::module& m) {
  py::class_<AttackActionConfig, ActionConfig, std::shared_ptr<AttackActionConfig>>(m, "AttackActionConfig")
      .def(py::init<const std::map<InventoryItem, InventoryQuantity>&,
                    const std::map<InventoryItem, InventoryProbability>&,
                    const std::map<InventoryItem, InventoryQuantity>&>(),
           py::arg("required_resources") = std::map<InventoryItem, InventoryQuantity>(),
           py::arg("consumed_resources") = std::map<InventoryItem, InventoryProbability>(),
           py::arg("defense_resources") = std::map<InventoryItem, InventoryQuantity>())
      .def_readwrite("defense_resources", &AttackActionConfig::defense_resources);
}

#endif  // PACKAGES_METTAGRID_CPP_INCLUDE_METTAGRID_ACTIONS_ATTACK_HPP_
