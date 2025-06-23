#ifndef ACTIONS_ATTACK_HPP_
#define ACTIONS_ATTACK_HPP_

#include <string>

#include "action_handler.hpp"
#include "grid_object.hpp"
#include "objects/agent.hpp"
#include "objects/constants.hpp"
#include "objects/metta_object.hpp"
#include "types.hpp"

class Attack : public ActionHandler {
public:
  explicit Attack(const ActionConfig& cfg, const std::string& action_name = "attack")
      : ActionHandler(cfg, action_name) {
    priority = 1;
  }

  unsigned char max_arg() const override {
    return 9;
  }

protected:
  bool _handle_action(Agent* actor, ActionArg arg) override {
    if (arg > 9 || arg < 1) {
      return false;
    }

    if (actor->update_inventory(InventoryItem::laser, -1) == 0) {
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
      actor->stats.incr("action." + _action_name + "." + ObjectTypeNames[agent_target->_type_id]);
      actor->stats.incr("action." + _action_name + "." + ObjectTypeNames[agent_target->_type_id] + "." +
                        actor->group_name);
      actor->stats.incr("action." + _action_name + "." + ObjectTypeNames[agent_target->_type_id] + "." +
                        actor->group_name + "." + agent_target->group_name);

      if (agent_target->group_name == actor->group_name) {
        actor->stats.incr("attack.own_team." + actor->group_name);
      } else {
        actor->stats.incr("attack.other_team." + actor->group_name);
      }

      was_frozen = agent_target->frozen > 0;

      if (agent_target->update_inventory(InventoryItem::armor, -1)) {
        actor->stats.incr("attack.blocked." + agent_target->group_name);
        actor->stats.incr("attack.blocked." + agent_target->group_name + "." + actor->group_name);
      } else {
        agent_target->frozen = agent_target->freeze_duration;

        if (!was_frozen) {
          // Actor (attacker) stats
          actor->stats.incr("attack.win." + actor->group_name);
          actor->stats.incr("attack.win." + actor->group_name + "." + agent_target->group_name);

          // Target (victim) stats - these should be on agent_target, not actor
          agent_target->stats.incr("attack.loss." + agent_target->group_name);
          agent_target->stats.incr("attack.loss." + agent_target->group_name + "." + actor->group_name);

          if (agent_target->group_name == actor->group_name) {
            actor->stats.incr("attack.win.own_team." + actor->group_name);
            agent_target->stats.incr("attack.loss.from_own_team." + agent_target->group_name);
          } else {
            actor->stats.incr("attack.win.other_team." + actor->group_name);
            agent_target->stats.incr("attack.loss.from_other_team." + agent_target->group_name);
          }

          for (int item = 0; item < InventoryItem::InventoryItemCount; item++) {
            int stolen = actor->update_inventory(static_cast<InventoryItem>(item), agent_target->inventory[item]);
            agent_target->update_inventory(static_cast<InventoryItem>(item), -stolen);
            if (stolen > 0) {
              actor->stats.add(InventoryItemNames[item] + ".stolen." + actor->group_name, stolen);
              // Also track what was stolen from the victim's perspective
              agent_target->stats.add(InventoryItemNames[item] + ".stolen_from." + agent_target->group_name, stolen);
            }
          }
        }

        return true;
      }
    }

    return false;
  }
};

#endif  // ACTIONS_ATTACK_HPP_
