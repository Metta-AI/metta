#ifndef ATTACK_HPP
#define ATTACK_HPP

#include <string>

#include "action_handler.hpp"
#include "grid_object.hpp"
#include "objects/agent.hpp"
#include "objects/constants.hpp"
#include "objects/metta_object.hpp"

class Attack : public ActionHandler {
public:
  Attack(const ActionConfig& cfg, const std::string& action_name = "attack") : ActionHandler(cfg, action_name) {
    priority = 1;
  }

  unsigned char max_arg() const override {
    return 9;
  }

protected:
  bool _handle_action(unsigned int actor_id, Agent* actor, ActionArg arg) override {
    if (arg > 9 || arg < 1) {
      return false;
    }

    if (actor->inventory[InventoryItem::laser] == 0) {
      return false;
    }

    actor->update_inventory(InventoryItem::laser, -1);

    short distance = 1 + (arg - 1) / 3;
    short offset = -((arg - 1) % 3 - 1);
    cout << "distance: " << distance << endl;
    cout << "offset: " << offset << endl;

    GridLocation target_loc =
        _grid->relative_location(actor->location, static_cast<Orientation>(actor->orientation), distance, offset);
    cout << "target_loc: " << target_loc.r << ", " << target_loc.c << ", " << target_loc.layer << endl;
    return _handle_target(actor_id, actor, target_loc);
  }

  bool _handle_target(unsigned int actor_id, Agent* actor, GridLocation target_loc) {
    target_loc.layer = GridLayer::Agent_Layer;
    cout << "target_loc: " << target_loc.r << ", " << target_loc.c << ", " << target_loc.layer << endl;
    cout << "grid: " << _grid << endl;
    Agent* agent_target = static_cast<Agent*>(_grid->object_at(target_loc));
    cout << "agent_target: " << agent_target << endl;
    bool was_frozen = false;
    if (agent_target) {
      cout << "agent_target found" << endl;
      actor->stats.incr(_stats.target[agent_target->_type_id]);
      actor->stats.incr(_stats.target[agent_target->_type_id], actor->group_name);
      actor->stats.incr(_stats.target[agent_target->_type_id], actor->group_name, agent_target->group_name);

      if (agent_target->group_name == actor->group_name) {
        actor->stats.incr("attack.own_team", actor->group_name);
      } else {
        actor->stats.incr("attack.other_team", actor->group_name);
      }

      was_frozen = agent_target->frozen > 0;

      if (agent_target->inventory[InventoryItem::armor] > 0) {
        cout << "agent_target has armor" << endl;
        agent_target->update_inventory(InventoryItem::armor, -1);
        actor->stats.incr("attack.blocked", agent_target->group_name);
        actor->stats.incr("attack.blocked", agent_target->group_name, actor->group_name);
      } else {
        cout << "agent_target does not have armor" << endl;
        agent_target->frozen = agent_target->freeze_duration;

        if (!was_frozen) {
          actor->stats.incr("attack.win", actor->group_name);
          actor->stats.incr("attack.win", actor->group_name, agent_target->group_name);
          actor->stats.incr("attack.loss", agent_target->group_name);
          actor->stats.incr("attack.loss", agent_target->group_name, actor->group_name);

          if (agent_target->group_name == actor->group_name) {
            actor->stats.incr("attack.win.own_team", actor->group_name);
          } else {
            actor->stats.incr("attack.win.other_team", actor->group_name);
          }

          for (int item = 0; item < InventoryItem::InventoryCount; item++) {
            actor->stats.add(InventoryItemNames[item], "stolen", actor->group_name, agent_target->inventory[item]);
            actor->update_inventory(static_cast<InventoryItem>(item), agent_target->inventory[item]);
            agent_target->update_inventory(static_cast<InventoryItem>(item), -agent_target->inventory[item]);
          }
        }

        return true;
      }
    }

    target_loc.layer = GridLayer::Object_Layer;
    cout << "target_loc: " << target_loc.r << ", " << target_loc.c << ", " << target_loc.layer << endl;
    MettaObject* object_target = static_cast<MettaObject*>(_grid->object_at(target_loc));
    cout << "object_target: " << object_target << endl;
    if (object_target) {
      cout << "object_target found" << endl;
      actor->stats.incr(_stats.target[object_target->_type_id]);
      actor->stats.incr(_stats.target[object_target->_type_id], actor->group_name);
      object_target->hp -= 1;
      actor->stats.incr("damage", ObjectTypeNames[object_target->_type_id]);
      if (object_target->hp <= 0) {
        cout << "object_target destroyed" << endl;
        _grid->remove_object(object_target);
        actor->stats.incr("destroyed", ObjectTypeNames[object_target->_type_id]);
      }
      return true;
    }

    return false;
  }
};

#endif  // ATTACK_HPP
