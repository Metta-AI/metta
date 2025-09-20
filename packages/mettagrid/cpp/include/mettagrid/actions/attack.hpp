#ifndef PACKAGES_METTAGRID_CPP_INCLUDE_METTAGRID_ACTIONS_ATTACK_HPP_
#define PACKAGES_METTAGRID_CPP_INCLUDE_METTAGRID_ACTIONS_ATTACK_HPP_

#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include <map>
#include <string>
#include <vector>

#include "actions/action_handler.hpp"
#include "core/grid_object.hpp"
#include "config/mettagrid_config.hpp"
#include "objects/agent.hpp"
#include "objects/constants.hpp"
#include "core/types.hpp"

// Attack takes an argument 0-8, which is the index of the target agent to attack.
// Target agents are those found in a 3x3 grid in front of the agent, indexed in scan order.
// If the argument (agent index to attack) > num_agents, the last agent is attacked.
struct AttackActionConfig : public ActionConfig {
  std::map<InventoryItem, InventoryQuantity> defense_resources;

  AttackActionConfig(const std::map<InventoryItem, InventoryQuantity>& required_resources,
                     const std::map<InventoryItem, InventoryProbability>& consumed_resources,
                     const std::map<InventoryItem, InventoryQuantity>& defense_resources)
      : ActionConfig(required_resources, consumed_resources), defense_resources(defense_resources) {}
};

class Attack : public ActionHandler {
public:
  explicit Attack(const AttackActionConfig& cfg,
                  const GameConfig* game_config,
                  const std::string& action_name = "attack")
      : ActionHandler(cfg, action_name), _defense_resources(cfg.defense_resources), _game_config(game_config) {
    priority = 1;
  }

  unsigned char max_arg() const override {
    return 8;
  }

protected:
  std::map<InventoryItem, InventoryQuantity> _defense_resources;
  const GameConfig* _game_config;

  bool _handle_action(Agent* actor, ActionArg arg) override {
    Agent* last_agent = nullptr;
    short num_skipped = 0;

    // Attack pattern depends on diagonal support
    if (!_game_config->allow_diagonals) {
      // Original 3x3 grid pattern for cardinal-only movement
      // 7 6 8  (3 cells forward)
      // 4 3 5  (2 cells forward)
      // 1 0 2  (1 cell forward)
      // . A .  (Agent position)

      static constexpr short COL_OFFSETS[3] = {0, -1, 1};

      for (short distance = 1; distance <= 3; distance++) {
        for (short offset : COL_OFFSETS) {
          GridLocation target_loc = _grid->relative_location(actor->location, actor->orientation, distance, offset);
          target_loc.layer = GridLayer::AgentLayer;

          Agent* target_agent = static_cast<Agent*>(_grid->object_at(target_loc));
          if (target_agent) {
            last_agent = target_agent;
            if (num_skipped == arg) {
              return _handle_target(*actor, *target_agent);
            }
            num_skipped++;
          }
        }
      }
    } else {
      // Diagonal attack pattern when diagonals are enabled
      // Agent facing NE example:
      // . 4 6 8
      // . 1 3 7
      // . 0 2 5
      // A . . .

      // Define the 9 positions in order (0-8)
      static constexpr struct {
        short forward;
        short lateral;
      } DIAGONAL_POSITIONS[9] = {
          {1, 0},   // 0: directly forward
          {2, -1},  // 1: forward-left
          {1, 1},   // 2: forward-right
          {2, 0},   // 3: 2 forward
          {3, -2},  // 4: far forward-left
          {1, 2},   // 5: right-forward
          {3, -1},  // 6: far forward-left-center
          {2, 1},   // 7: forward-right-center
          {3, 0}    // 8: 3 forward
      };

      for (const auto& pos : DIAGONAL_POSITIONS) {
        GridLocation target_loc =
            _grid->relative_location(actor->location, actor->orientation, pos.forward, pos.lateral);
        target_loc.layer = GridLayer::AgentLayer;

        Agent* target_agent = static_cast<Agent*>(_grid->object_at(target_loc));
        if (target_agent) {
          last_agent = target_agent;
          if (num_skipped == arg) {
            return _handle_target(*actor, *target_agent);
          }
          num_skipped++;
        }
      }
    }

    // If we got here, it means we skipped over all the targets. Attack the last one.
    if (last_agent) {
      return _handle_target(*actor, *last_agent);
    }

    return false;
  }

  bool _handle_target(Agent& actor, Agent& target) {
    bool was_already_frozen = target.frozen > 0;

    // Check if target can defend
    if (!_defense_resources.empty()) {
      bool target_can_defend = _check_defense_capability(target);

      if (target_can_defend) {
        _consume_defense_resources(target);
        _log_blocked_attack(actor, target);
        return true;
      }
    }

    // Attack succeeds
    target.frozen = target.freeze_duration;

    if (!was_already_frozen) {
      _steal_resources(actor, target);
      _log_successful_attack(actor, target);
    } else {
      // Track wasted attacks on already-frozen targets
      const std::string& actor_group = actor.group_name;
      actor.stats.incr(_action_prefix(actor_group) + "wasted_on_frozen");
    }
    return true;
  }

private:
  bool _check_defense_capability(const Agent& target) const {
    for (const auto& [item, amount] : _defense_resources) {
      auto it = target.inventory.find(item);
      if (it == target.inventory.end() || it->second < amount) {
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
    // Create snapshot to avoid iterator invalidation
    std::vector<std::pair<InventoryItem, InventoryQuantity>> snapshot;
    snapshot.reserve(target.inventory.size());
    for (const auto& [item, amount] : target.inventory) {
      snapshot.emplace_back(item, amount);
    }

    // Transfer resources
    for (const auto& [item, amount] : snapshot) {
      InventoryDelta stolen = actor.update_inventory(item, amount);
      target.update_inventory(item, -stolen);

      if (stolen > 0) {
        _log_resource_theft(actor, target, item, stolen);
      }
    }
  }

  std::string _action_prefix(const std::string& group) const {
    return "action." + _action_name + "." + group + ".";
  }

  void _log_blocked_attack(Agent& actor, const Agent& target) const {
    const std::string& actor_group = actor.group_name;
    const std::string& target_group = target.group_name;

    actor.stats.incr(_action_prefix(actor_group) + "blocked_by." + target_group);
  }

  void _log_successful_attack(Agent& actor, Agent& target) const {
    const std::string& actor_group = actor.group_name;
    const std::string& target_group = target.group_name;
    bool same_team = (actor_group == target_group);

    if (same_team) {
      actor.stats.incr(_action_prefix(actor_group) + "friendly_fire");
    } else {
      actor.stats.incr(_action_prefix(actor_group) + "hit." + target_group);
      target.stats.incr(_action_prefix(target_group) + "hit_by." + actor_group);
    }
  }

  void _log_resource_theft(Agent& actor, Agent& target, InventoryItem item, InventoryDelta amount) const {
    const std::string& actor_group = actor.group_name;
    const std::string& target_group = target.group_name;
    const std::string item_name = actor.stats.resource_name(item);

    actor.stats.add(_action_prefix(actor_group) + "steals." + item_name + ".from." + target_group, amount);
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
