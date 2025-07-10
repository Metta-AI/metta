#ifndef ACTIONS_ATTACK_HPP_
#define ACTIONS_ATTACK_HPP_

#include <map>
#include <string>
#include <vector>

#include "action_handler.hpp"
#include "grid_object.hpp"
#include "objects/agent.hpp"
#include "objects/constants.hpp"
#include "types.hpp"

// Attack takes an argument 0-8, which is the index of the target agent to attack.
// Target agents are those found in a 3x3 grid in front of the agent, indexed in scan order.
// If the argument (agent index to attack) > num_agents, the last agent is attacked.
struct AttackActionConfig : public ActionConfig {
  std::map<InventoryItem, InventoryQuantity> defense_resources;

  AttackActionConfig(const std::map<InventoryItem, InventoryQuantity>& required_resources,
                     const std::map<InventoryItem, InventoryQuantity>& consumed_resources,
                     const std::map<InventoryItem, InventoryQuantity>& defense_resources)
      : ActionConfig(required_resources, consumed_resources), defense_resources(defense_resources) {}
};

class Attack : public ActionHandler {
public:
  explicit Attack(const AttackActionConfig& cfg, const std::string& action_name = "attack")
      : ActionHandler(cfg, action_name), _defense_resources(cfg.defense_resources) {
    priority = 1;
  }

  unsigned char max_arg() const override {
    return 8;
  }

protected:
  std::map<InventoryItem, InventoryQuantity> _defense_resources;

  bool _handle_action(Agent* actor, ActionArg arg) override {
    if (arg > 8 || arg < 0) {
      return false;
    }

    short num_found = 0;
    Agent* last_agent = nullptr;

    // Attack positions form a 3x3 grid in front of the agent
    // Visual representation of attack order (agent facing up):
    // 7 6 8  (3 cells forward)
    // 4 3 5  (2 cells forward)
    // 1 0 2  (1 cell forward)
    //   A    (Agent position)
    static constexpr std::pair<short, short> ATTACK_POSITIONS[9] = {
        {1, 0},   // 0: 1 forward, straight ahead
        {1, -1},  // 1: 1 forward, 1 left
        {1, 1},   // 2: 1 forward, 1 right
        {2, 0},   // 3: 2 forward, straight ahead
        {2, -1},  // 4: 2 forward, 1 left
        {2, 1},   // 5: 2 forward, 1 right
        {3, 0},   // 6: 3 forward, straight ahead
        {3, -1},  // 7: 3 forward, 1 left
        {3, 1},   // 8: 3 forward, 1 right
    };

    // Scan the 9 squares in front of the agent (3x3 grid)
    for (int i = 0; i < 9; ++i) {
      auto [distance, offset] = ATTACK_POSITIONS[i];
      GridLocation target_loc = _grid->relative_location(actor->location, actor->orientation, distance, offset);
      target_loc.layer = GridLayer::AgentLayer;

      Agent* agent_target = static_cast<Agent*>(_grid->object_at(target_loc));
      if (agent_target != nullptr) {
        last_agent = agent_target;

        // If this is the agent at the requested index, attack it
        if (num_found == arg) {
          return _handle_target(*actor, *agent_target);
        }
        num_found++;
      }
    }

    // If we got here, it means we skipped over all the targets. Attack the last one.
    if (last_agent != nullptr && arg >= num_found) {
      return _handle_target(*actor, *last_agent);
    }

    return false;
  }

  bool _handle_target(Agent& actor, Agent& target) {
    bool was_frozen = false;

    // Track attack targets
    actor.stats.incr("action." + _action_name + "." + target.type_name);
    actor.stats.incr("action." + _action_name + "." + target.type_name + "." + actor.group_name);
    actor.stats.incr("action." + _action_name + "." + target.type_name + "." + actor.group_name + "." +
                     target.group_name);

    if (target.group_name == actor.group_name) {
      actor.stats.incr("attack.own_team." + actor.group_name);
    } else {
      actor.stats.incr("attack.other_team." + actor.group_name);
    }

    was_frozen = target.frozen > 0;

    bool blocked = _defense_resources.size() > 0;
    for (const auto& [item, amount] : _defense_resources) {
      if (target.inventory[item] < amount) {
        blocked = false;
        break;
      }
    }

    if (blocked) {
      // Consume the defense resources
      for (const auto& [item, amount] : _defense_resources) {
        int used_amount = std::abs(target.update_inventory(item, -amount));
        assert(used_amount == amount);
      }

      actor.stats.incr("attack.blocked." + target.group_name);
      actor.stats.incr("attack.blocked." + target.group_name + "." + actor.group_name);
      return true;
    }

    target.frozen = target.freeze_duration;

    if (!was_frozen) {
      // Actor (attacker) stats
      actor.stats.incr("attack.win." + actor.group_name);
      actor.stats.incr("attack.win." + actor.group_name + "." + target.group_name);

      // Target (victim) stats - these should be on agent_target, not actor
      target.stats.incr("attack.loss." + target.group_name);
      target.stats.incr("attack.loss." + target.group_name + "." + actor.group_name);

      if (target.group_name == actor.group_name) {
        actor.stats.incr("attack.win.own_team." + actor.group_name);
        target.stats.incr("attack.loss.from_own_team." + target.group_name);
      } else {
        actor.stats.incr("attack.win.other_team." + actor.group_name);
        target.stats.incr("attack.loss.from_other_team." + target.group_name);
      }

      // Collect all items to steal first, then apply changes, since the changes
      // can delete keys from the agent's inventory.
      std::vector<std::pair<InventoryItem, int>> resources_to_steal;
      for (const auto& [item, amount] : target.inventory) {
        resources_to_steal.emplace_back(item, amount);
      }

      // Now apply the stealing
      for (const auto& [item, amount] : resources_to_steal) {
        InventoryDelta stolen = actor.update_inventory(item, static_cast<InventoryDelta>(amount));

        target.update_inventory(item, -stolen);
        if (stolen > 0) {
          actor.stats.add(actor.stats.inventory_item_name(item) + ".stolen." + actor.group_name, stolen);
          // Also track what was stolen from the victim's perspective
          target.stats.add(target.stats.inventory_item_name(item) + ".stolen_from." + target.group_name, stolen);
        }
      }
    }

    return true;
  }
};

#endif  // ACTIONS_ATTACK_HPP_
