#ifndef PACKAGES_METTAGRID_CPP_INCLUDE_METTAGRID_ACTIONS_ATTACK_HPP_
#define PACKAGES_METTAGRID_CPP_INCLUDE_METTAGRID_ACTIONS_ATTACK_HPP_

#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include <algorithm>
#include <cassert>
#include <optional>
#include <string>
#include <unordered_map>
#include <vector>

#include "actions/action_handler.hpp"
#include "config/mettagrid_config.hpp"
#include "core/grid_object.hpp"
#include "core/types.hpp"
#include "objects/agent.hpp"
#include "objects/constants.hpp"

// Attack takes an argument 0-8, which is the index of the target agent to attack.
// Target agents are those found in a 3x3 grid in front of the agent, indexed in scan order.
// If the argument (agent index to attack) > num_agents, the last agent is attacked.
struct AttackActionConfig : public ActionConfig {
  std::unordered_map<InventoryItem, InventoryQuantity> defense_resources;
  std::unordered_map<InventoryItem, InventoryQuantity> armor_resources;
  std::unordered_map<InventoryItem, InventoryQuantity> weapon_resources;
  std::optional<std::vector<InventoryItem>> loot;
  bool enabled;
  std::vector<ObservationType> vibes;  // Vibes that trigger this action on move

  AttackActionConfig(const std::unordered_map<InventoryItem, InventoryQuantity>& required_resources,
                     const std::unordered_map<InventoryItem, InventoryQuantity>& consumed_resources,
                     const std::unordered_map<InventoryItem, InventoryQuantity>& defense_resources,
                     const std::unordered_map<InventoryItem, InventoryQuantity>& armor_resources,
                     const std::unordered_map<InventoryItem, InventoryQuantity>& weapon_resources,
                     const std::optional<std::vector<InventoryItem>>& loot = std::nullopt,
                     bool enabled = true,
                     const std::vector<ObservationType>& vibes = {})
      : ActionConfig(required_resources, consumed_resources),
        defense_resources(defense_resources),
        armor_resources(armor_resources),
        weapon_resources(weapon_resources),
        loot(loot),
        enabled(enabled),
        vibes(vibes) {}
};

class Attack : public ActionHandler {
public:
  explicit Attack(const AttackActionConfig& cfg,
                  const GameConfig* game_config,
                  const std::string& action_name = "attack")
      : ActionHandler(cfg, action_name),
        _defense_resources(cfg.defense_resources),
        _armor_resources(cfg.armor_resources),
        _weapon_resources(cfg.weapon_resources),
        _loot(cfg.loot),
        _game_config(game_config),
        _enabled(cfg.enabled),
        _vibes(cfg.vibes) {
    priority = 1;
  }

  // Get vibes that trigger this action on move
  const std::vector<ObservationType>& get_vibes() const {
    return _vibes;
  }

  std::vector<Action> create_actions() override {
    std::vector<Action> actions;
    if (_enabled) {
      // Attack is enabled - create actions
      for (unsigned char i = 0; i <= 8; ++i) {
        actions.emplace_back(this, "attack_" + std::to_string(i), static_cast<ActionArg>(i));
      }
    }
    return actions;
  }

  // Expose to Move class - attack decides if target is valid
  bool try_attack(Agent& actor, GridObject* target_object) {
    if (!target_object) return false;

    Agent* target = dynamic_cast<Agent*>(target_object);
    if (!target) return false;  // Can only attack agents

    // Check if actor has required resources for attack
    for (const auto& [item, amount] : _consumed_resources) {
      if (actor.inventory.amount(item) < amount) {
        return false;  // Can't afford attack
      }
    }

    bool success = _handle_target(actor, *target);

    // Consume resources on success
    if (success) {
      for (const auto& [item, amount] : _consumed_resources) {
        if (amount > 0) {
          InventoryDelta delta = static_cast<InventoryDelta>(-static_cast<int>(amount));
          actor.inventory.update(item, delta);
        }
      }
    }

    return success;
  }

protected:
  std::unordered_map<InventoryItem, InventoryQuantity> _defense_resources;
  std::unordered_map<InventoryItem, InventoryQuantity> _armor_resources;
  std::unordered_map<InventoryItem, InventoryQuantity> _weapon_resources;
  std::optional<std::vector<InventoryItem>> _loot;
  const GameConfig* _game_config;
  bool _enabled;
  std::vector<ObservationType> _vibes;

  bool _handle_action(Agent& actor, ActionArg arg) override {
    std::vector<Agent*> targets;
    // Simple 3x3 neighborhood scan
    for (int r = static_cast<int>(actor.location.r) - 1; r <= static_cast<int>(actor.location.r) + 1; r++) {
      for (int c = static_cast<int>(actor.location.c) - 1; c <= static_cast<int>(actor.location.c) + 1; c++) {
        if (r == actor.location.r && c == actor.location.c) continue;

        GridLocation loc(static_cast<GridCoord>(r), static_cast<GridCoord>(c));
        Agent* agent = dynamic_cast<Agent*>(_grid->object_at(loc));
        if (agent) {
          targets.push_back(agent);
        }
      }
    }

    if (!targets.empty()) {
      // "If the argument > num_agents, the last agent is attacked."
      if (static_cast<size_t>(arg) >= targets.size()) {
        return _handle_target(actor, *targets.back());
      } else {
        return _handle_target(actor, *targets[arg]);
      }
    }

    return false;
  }

  bool _handle_target(Agent& actor, Agent& target) {
    bool was_already_frozen = target.frozen > 0;

    // Check if target can defend
    if (!_defense_resources.empty() || !_armor_resources.empty() || !_weapon_resources.empty()) {
      bool target_can_defend = _check_defense_capability(actor, target);

      if (target_can_defend) {
        _consume_defense_resources(actor, target);
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
  // Compute weapon power from attacker's inventory
  int _compute_weapon_power(const Agent& attacker) const {
    int power = 0;
    for (const auto& [item, weight] : _weapon_resources) {
      power += attacker.inventory.amount(item) * weight;
    }
    return power;
  }

  // Check if target is vibing a specific resource (gets +1 effective armor)
  bool _is_vibing_resource(const Agent& target, InventoryItem item) const {
    assert(_game_config && "Attack handler must have valid game_config pointer");
    if (target.vibe == 0 || target.vibe >= _game_config->vibe_names.size() ||
        item >= _game_config->resource_names.size()) {
      return false;
    }
    return _game_config->vibe_names[target.vibe] == _game_config->resource_names[item];
  }

  // Compute armor power from target's inventory
  // Vibing an armor resource counts as having +1 of that resource
  int _compute_armor_power(const Agent& target) const {
    int power = 0;
    for (const auto& [item, weight] : _armor_resources) {
      int amount = target.inventory.amount(item);
      if (_is_vibing_resource(target, item)) {
        amount += 1;
      }
      power += amount * weight;
    }
    return power;
  }

  bool _check_defense_capability(const Agent& attacker, const Agent& target) const {
    // Compute weapon vs armor difference
    int weapon_power = _compute_weapon_power(attacker);
    int armor_power = _compute_armor_power(target);

    // Damage bonus: max(weapon_power - armor_power, 0)
    // Weapon power increases defense cost needed
    int damage_bonus = std::max(weapon_power - armor_power, 0);

    // Check defense resources (increased by damage bonus)
    for (const auto& [item, amount] : _defense_resources) {
      // required = defense_resources + max(weapon - armor, 0)
      int required = static_cast<int>(amount) + damage_bonus;
      InventoryQuantity has = target.inventory.amount(item);
      if (has < static_cast<InventoryQuantity>(required)) {
        return false;
      }
    }
    return true;
  }

  void _consume_defense_resources(const Agent& attacker, Agent& target) {
    int weapon_power = _compute_weapon_power(attacker);
    int armor_power = _compute_armor_power(target);
    int damage_bonus = std::max(weapon_power - armor_power, 0);

    for (const auto& [item, amount] : _defense_resources) {
      int required = static_cast<int>(amount) + damage_bonus;
      [[maybe_unused]] InventoryDelta delta = target.inventory.update(item, -required);
    }
  }

  void _steal_resources(Agent& actor, Agent& target) {
    // Identify resources to steal
    std::vector<std::pair<InventoryItem, InventoryQuantity>> resources_to_steal;

    if (_loot.has_value()) {
      // Steal only listed resources (in specified order)
      for (const auto& item : *_loot) {
        InventoryQuantity amount = target.inventory.amount(item);
        if (amount > 0) {
          resources_to_steal.emplace_back(item, amount);
        }
      }
    } else {
      // Steal everything (backward compatibility)
      resources_to_steal.reserve(target.inventory.get().size());
      for (const auto& [item, amount] : target.inventory.get()) {
        resources_to_steal.emplace_back(item, amount);
      }
    }

    // Transfer resources
    for (const auto& [item, amount] : resources_to_steal) {
      InventoryDelta stolen = actor.inventory.update(item, amount);
      target.inventory.update(item, -stolen);

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
      .def(py::init<const std::unordered_map<InventoryItem, InventoryQuantity>&,
                    const std::unordered_map<InventoryItem, InventoryQuantity>&,
                    const std::unordered_map<InventoryItem, InventoryQuantity>&,
                    const std::unordered_map<InventoryItem, InventoryQuantity>&,
                    const std::unordered_map<InventoryItem, InventoryQuantity>&,
                    const std::optional<std::vector<InventoryItem>>&,
                    bool,
                    const std::vector<ObservationType>&>(),
           py::arg("required_resources") = std::unordered_map<InventoryItem, InventoryQuantity>(),
           py::arg("consumed_resources") = std::unordered_map<InventoryItem, InventoryQuantity>(),
           py::arg("defense_resources") = std::unordered_map<InventoryItem, InventoryQuantity>(),
           py::arg("armor_resources") = std::unordered_map<InventoryItem, InventoryQuantity>(),
           py::arg("weapon_resources") = std::unordered_map<InventoryItem, InventoryQuantity>(),
           py::arg("loot") = std::nullopt,
           py::arg("enabled") = true,
           py::arg("vibes") = std::vector<ObservationType>())
      .def_readwrite("defense_resources", &AttackActionConfig::defense_resources)
      .def_readwrite("armor_resources", &AttackActionConfig::armor_resources)
      .def_readwrite("weapon_resources", &AttackActionConfig::weapon_resources)
      .def_readwrite("loot", &AttackActionConfig::loot)
      .def_readwrite("enabled", &AttackActionConfig::enabled)
      .def_readwrite("vibes", &AttackActionConfig::vibes);
}

#endif  // PACKAGES_METTAGRID_CPP_INCLUDE_METTAGRID_ACTIONS_ATTACK_HPP_
