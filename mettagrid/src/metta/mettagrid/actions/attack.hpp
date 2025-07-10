#ifndef ACTIONS_ATTACK_HPP_
#define ACTIONS_ATTACK_HPP_

#include <string>
#include <vector>

#include "action_handler.hpp"
#include "grid_object.hpp"
#include "objects/agent.hpp"
#include "objects/constants.hpp"
#include "objects/metta_object.hpp"
#include "types.hpp"

struct AttackActionConfig : public ActionConfig {
  std::map<InventoryItem, int> defense_resources;

  AttackActionConfig(const std::map<InventoryItem, int>& required_resources,
                     const std::map<InventoryItem, int>& consumed_resources,
                     const std::map<InventoryItem, int>& defense_resources)
      : ActionConfig(required_resources, consumed_resources), defense_resources(defense_resources) {}
};

class Attack : public ActionHandler {
public:
  explicit Attack(const AttackActionConfig& cfg, const std::string& action_name = "attack")
      : ActionHandler(cfg, action_name), _defense_resources(cfg.defense_resources) {
    priority = 1;
  }

  unsigned char max_arg() const override {
    return 9;
  }

protected:
  std::map<InventoryItem, int> _defense_resources;

  bool _handle_action(Agent* actor, ActionArg arg) override {
    if (arg > 9 || arg < 1) {
      return false;
    }

    short distance = 1 + (arg - 1) / 3;
    short offset = -((arg - 1) % 3 - 1);

    GridLocation target_loc =
        _grid->relative_location(actor->location, static_cast<Orientation>(actor->orientation), distance, offset);

    return _handle_target(actor, target_loc);
  }

  bool _handle_target(Agent* actor, GridLocation target_loc) {
    target_loc.layer = GridLayer::Agent_Layer;
    Agent* agent_target = static_cast<Agent*>(_grid->object_at(target_loc));

    bool was_frozen = false;
    if (agent_target) {
      // Track attack targets
      actor->stats.incr("action." + _action_name + "." + agent_target->type_name);
      actor->stats.incr("action." + _action_name + "." + agent_target->type_name + "." + actor->group_name);
      actor->stats.incr("action." + _action_name + "." + agent_target->type_name + "." + actor->group_name + "." +
                        agent_target->group_name);

      if (agent_target->group_name == actor->group_name) {
        actor->stats.incr("attack.own_team." + actor->group_name);
      } else {
        actor->stats.incr("attack.other_team." + actor->group_name);
      }

      bool was_already_frozen = target->frozen > 0;

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
      target->frozen = target->freeze_duration;

      if (!was_already_frozen) {
        _steal_resources(actor, target);
        _log_successful_attack(actor, target);
      } else {
        // Track wasted attacks on already-frozen targets
        actor->stats.incr("action." + _action_name + ".wasted_on_frozen");
      }
      return true;
    }

  private:
    bool _check_defense_capability(const Agent* target) const {
      for (const auto& [item, amount] : _defense_resources) {
        auto it = target->inventory.find(item);
        if (it == target->inventory.end() || it->second < amount) {
          return false;
        }
      }
      return true;
    }

    void _consume_defense_resources(Agent * target) {
      for (const auto& [item, amount] : _defense_resources) {
        InventoryDelta used = std::abs(target->update_inventory(item, -static_cast<InventoryDelta>(amount)));
        assert(used == static_cast<InventoryDelta>(amount));
      }
    }

    void _steal_resources(Agent * actor, Agent * target) {
      // Create snapshot to avoid iterator invalidation
      std::vector<std::pair<InventoryItem, InventoryQuantity>> snapshot;
      for (const auto& [item, amount] : target->inventory) {
        snapshot.emplace_back(item, amount);
      }

      // Transfer resources
      for (const auto& [item, amount] : snapshot) {
        InventoryDelta stolen = actor->update_inventory(item, static_cast<InventoryDelta>(amount));
        target->update_inventory(item, -stolen);

        if (stolen > 0) {
          _log_resource_theft(actor, target, item, stolen);
        }
      }
    }

    void _log_blocked_attack(Agent * actor, const Agent* target) const {
      const std::string& actor_group = actor->group_name;
      const std::string& target_group = target->group_name;

      // Just log that this actor's attack was blocked by this target group
      actor->stats.incr("action." + _action_name + ".blocked_by." + target_group);
    }

    void _log_successful_attack(Agent * actor, Agent * target) const {
      const std::string& actor_group = actor->group_name;
      const std::string& target_group = target->group_name;
      bool same_team = (actor_group == target_group);

      // Key stats: who did you successfully attack and was it friendly fire?
      if (same_team) {
        actor->stats.incr("action." + _action_name + ".friendly_fire");
        target->stats.incr("action." + _action_name + ".victim_of_friendly_fire");
      } else {
        actor->stats.incr("action." + _action_name + ".hits." + target_group);
        target->stats.incr("action." + _action_name + ".hit_by." + actor_group);
      }
    }

    void _log_resource_theft(Agent * actor, Agent * target, InventoryItem item, InventoryDelta amount) const {
      const std::string& actor_group = actor->group_name;
      const std::string& target_group = target->group_name;
      const std::string item_name = actor->stats.inventory_item_name(item);

      // Just track total resources stolen between groups
      actor->stats.add(item_name + ".stolen_from." + target_group, amount);
    }
  };

#endif  // ACTIONS_ATTACK_HPP_
