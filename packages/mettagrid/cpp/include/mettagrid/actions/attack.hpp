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

// Outcome configuration for attack success
struct AttackOutcome {
  std::unordered_map<InventoryItem, InventoryDelta> actor_inv_delta;   // Inventory changes for attacker
  std::unordered_map<InventoryItem, InventoryDelta> target_inv_delta;  // Inventory changes for target
  std::vector<InventoryItem> loot;                                     // Resources to steal from target
  int freeze;                                                          // Freeze duration (0 = no freeze)

  AttackOutcome(const std::unordered_map<InventoryItem, InventoryDelta>& actor_inv_delta = {},
                const std::unordered_map<InventoryItem, InventoryDelta>& target_inv_delta = {},
                const std::vector<InventoryItem>& loot = {},
                int freeze = 0)
      : actor_inv_delta(actor_inv_delta), target_inv_delta(target_inv_delta), loot(loot), freeze(freeze) {}
};

// Attack is triggered by moving onto another agent (when vibes match).
// No standalone attack actions are created - attack only happens via move.
struct AttackActionConfig : public ActionConfig {
  std::unordered_map<InventoryItem, InventoryQuantity> defense_resources;
  std::unordered_map<InventoryItem, InventoryQuantity> armor_resources;
  std::unordered_map<InventoryItem, InventoryQuantity> weapon_resources;
  AttackOutcome success;  // Outcome when attack succeeds
  bool enabled;
  std::vector<ObservationType> vibes;                   // Vibes that trigger this action on move
  std::unordered_map<ObservationType, int> vibe_bonus;  // Per-vibe armor bonus

  AttackActionConfig(const std::unordered_map<InventoryItem, InventoryQuantity>& required_resources,
                     const std::unordered_map<InventoryItem, InventoryQuantity>& consumed_resources,
                     const std::unordered_map<InventoryItem, InventoryQuantity>& defense_resources,
                     const std::unordered_map<InventoryItem, InventoryQuantity>& armor_resources,
                     const std::unordered_map<InventoryItem, InventoryQuantity>& weapon_resources,
                     const AttackOutcome& success = AttackOutcome(),
                     bool enabled = true,
                     const std::vector<ObservationType>& vibes = {},
                     const std::unordered_map<ObservationType, int>& vibe_bonus = {})
      : ActionConfig(required_resources, consumed_resources),
        defense_resources(defense_resources),
        armor_resources(armor_resources),
        weapon_resources(weapon_resources),
        success(success),
        enabled(enabled),
        vibes(vibes),
        vibe_bonus(vibe_bonus) {}
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
        _success(cfg.success),
        _game_config(game_config),
        _enabled(cfg.enabled),
        _vibes(cfg.vibes),
        _vibe_bonus(cfg.vibe_bonus) {
    priority = 1;
  }

  // Get vibes that trigger this action on move
  const std::vector<ObservationType>& get_vibes() const {
    return _vibes;
  }

  std::vector<Action> create_actions() override {
    // Attack only triggers via move, no standalone actions
    return {};
  }

  // Expose to Move class - attack decides if target is valid
  bool try_attack(Agent& actor, GridObject* target_object) {
    if (!target_object) return false;

    Agent* target = dynamic_cast<Agent*>(target_object);
    if (!target) return false;  // Can only attack agents

    // Don't attack already frozen agents - let move handler swap instead
    if (target->frozen > 0) return false;

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
  AttackOutcome _success;
  const GameConfig* _game_config;
  bool _enabled;
  std::vector<ObservationType> _vibes;
  std::unordered_map<ObservationType, int> _vibe_bonus;

  bool _handle_action(Agent& /*actor*/, ActionArg /*arg*/) override {
    // Attack only triggers via move onto target, not as standalone action
    return false;
  }

  bool _handle_target(Agent& actor, Agent& target) {
    // Check if target can defend (requires defense_resources to be configured)
    // armor/weapon resources only modify the defense cost, they don't enable defense by themselves
    if (!_defense_resources.empty()) {
      bool target_can_defend = _check_defense_capability(actor, target);

      if (target_can_defend) {
        _consume_defense_resources(actor, target);
        _log_blocked_attack(actor, target);
        return true;
      }
    }

    // Attack succeeds - apply configured outcome
    if (_success.freeze > 0) {
      target.frozen = _success.freeze;
    }

    _apply_outcome(actor, target);
    _log_successful_attack(actor, target);
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

  // Get vibe bonus for target's current vibe (0 if not configured)
  int _get_vibe_bonus(const Agent& target) const {
    auto it = _vibe_bonus.find(target.vibe);
    return (it != _vibe_bonus.end()) ? it->second : 0;
  }

  // Check if target is vibing a specific resource
  bool _is_vibing_resource(const Agent& target, InventoryItem item) const {
    assert(_game_config && "Attack handler must have valid game_config pointer");
    if (target.vibe == 0 || target.vibe >= _game_config->vibe_names.size() ||
        item >= _game_config->resource_names.size()) {
      return false;
    }
    return _game_config->vibe_names[target.vibe] == _game_config->resource_names[item];
  }

  // Compute armor power from target's inventory
  // Vibing an armor resource adds the per-vibe bonus for that resource
  int _compute_armor_power(const Agent& target) const {
    int power = 0;
    for (const auto& [item, weight] : _armor_resources) {
      int amount = target.inventory.amount(item);
      if (_is_vibing_resource(target, item)) {
        amount += _get_vibe_bonus(target);
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

  void _apply_outcome(Agent& actor, Agent& target) {
    // Apply actor inventory changes
    for (const auto& [item, delta] : _success.actor_inv_delta) {
      actor.inventory.update(item, delta);
    }

    // Apply target inventory changes
    for (const auto& [item, delta] : _success.target_inv_delta) {
      target.inventory.update(item, delta);
    }

    // Steal loot from target
    for (const auto& item : _success.loot) {
      InventoryQuantity amount = target.inventory.amount(item);
      if (amount > 0) {
        InventoryDelta stolen = actor.inventory.update(item, amount);
        target.inventory.update(item, -stolen);
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
};

namespace py = pybind11;

inline void bind_attack_action_config(py::module& m) {
  py::class_<AttackOutcome, std::shared_ptr<AttackOutcome>>(m, "AttackOutcome")
      .def(py::init<const std::unordered_map<InventoryItem, InventoryDelta>&,
                    const std::unordered_map<InventoryItem, InventoryDelta>&,
                    const std::vector<InventoryItem>&,
                    int>(),
           py::arg("actor") = std::unordered_map<InventoryItem, InventoryDelta>(),
           py::arg("target") = std::unordered_map<InventoryItem, InventoryDelta>(),
           py::arg("loot") = std::vector<InventoryItem>(),
           py::arg("freeze") = 0)
      .def_readwrite("actor_inv_delta", &AttackOutcome::actor_inv_delta)
      .def_readwrite("target_inv_delta", &AttackOutcome::target_inv_delta)
      .def_readwrite("loot", &AttackOutcome::loot)
      .def_readwrite("freeze", &AttackOutcome::freeze);

  py::class_<AttackActionConfig, ActionConfig, std::shared_ptr<AttackActionConfig>>(m, "AttackActionConfig")
      .def(py::init<const std::unordered_map<InventoryItem, InventoryQuantity>&,
                    const std::unordered_map<InventoryItem, InventoryQuantity>&,
                    const std::unordered_map<InventoryItem, InventoryQuantity>&,
                    const std::unordered_map<InventoryItem, InventoryQuantity>&,
                    const std::unordered_map<InventoryItem, InventoryQuantity>&,
                    const AttackOutcome&,
                    bool,
                    const std::vector<ObservationType>&,
                    const std::unordered_map<ObservationType, int>&>(),
           py::arg("required_resources") = std::unordered_map<InventoryItem, InventoryQuantity>(),
           py::arg("consumed_resources") = std::unordered_map<InventoryItem, InventoryQuantity>(),
           py::arg("defense_resources") = std::unordered_map<InventoryItem, InventoryQuantity>(),
           py::arg("armor_resources") = std::unordered_map<InventoryItem, InventoryQuantity>(),
           py::arg("weapon_resources") = std::unordered_map<InventoryItem, InventoryQuantity>(),
           py::arg("success") = AttackOutcome(),
           py::arg("enabled") = true,
           py::arg("vibes") = std::vector<ObservationType>(),
           py::arg("vibe_bonus") = std::unordered_map<ObservationType, int>())
      .def_readwrite("defense_resources", &AttackActionConfig::defense_resources)
      .def_readwrite("armor_resources", &AttackActionConfig::armor_resources)
      .def_readwrite("weapon_resources", &AttackActionConfig::weapon_resources)
      .def_readwrite("success", &AttackActionConfig::success)
      .def_readwrite("enabled", &AttackActionConfig::enabled)
      .def_readwrite("vibes", &AttackActionConfig::vibes)
      .def_readwrite("vibe_bonus", &AttackActionConfig::vibe_bonus);
}

#endif  // PACKAGES_METTAGRID_CPP_INCLUDE_METTAGRID_ACTIONS_ATTACK_HPP_
