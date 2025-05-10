#ifndef ATTACK_HPP
#define ATTACK_HPP

#include <cstdint>
#include <string>

#include "actions/action_handler.hpp"
#include "constants.hpp"
#include "objects/agent.hpp"
#include "objects/metta_object.hpp"
namespace Actions {
class Attack : public ActionHandler {
public:
  Attack(const ActionConfig& cfg, const std::string& action_name = "attack") : ActionHandler(cfg, action_name) {
    priority = 1;
  }

  uint8_t max_arg() const override {
    return 9;
  }

  ActionHandler* clone() const override {
    return new Attack(*this);
  }

protected:
  bool _handle_action(uint32_t actor_id, Agent* actor, c_actions_type arg) override {
    // Validate action argument
    if (arg > 9 || arg < 1) {
      return false;  // Invalid arg is a normal gameplay situation
    }

    // Check if agent has lasers
    if (actor->inventory[InventoryItem::laser] == 0) {
      return false;  // No lasers is a normal gameplay situation
    }

    // Consume a laser
    actor->update_inventory(InventoryItem::laser, -1);

    // Calculate target location
    int16_t distance = 1 + (arg - 1) / 3;
    int16_t offset = -((arg - 1) % 3 - 1);

    // Use utility method from base class
    validate_orientation(actor);

    GridLocation target_loc =
        _grid->relative_location(actor->location, static_cast<Orientation>(actor->orientation), distance, offset);

    return _handle_target(actor_id, actor, target_loc);
  }

  bool _handle_target(uint32_t actor_id, Agent* actor, GridLocation target_loc) {
    // Check if target location is within grid bounds using base class utility
    if (!is_valid_location(target_loc)) {
      return false;  // Out of bounds is a normal gameplay situation
    }

    // Try to find an agent at the target location using safe_object_at
    target_loc.layer = GridLayer::Agent_Layer;
    GridObject* obj = safe_object_at(target_loc);

    // If we find an object, make sure it's an agent
    Agent* agent_target = nullptr;
    if (obj != nullptr) {
      agent_target = dynamic_cast<Agent*>(obj);
      if (obj != nullptr && agent_target == nullptr) {
        throw std::runtime_error("Object at target location is not an Agent");
      }
    }

    bool was_frozen = false;
    if (agent_target) {
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
        agent_target->update_inventory(InventoryItem::armor, -1);
        actor->stats.incr("attack.blocked", agent_target->group_name);
        actor->stats.incr("attack.blocked", agent_target->group_name, actor->group_name);
      } else {
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

          for (size_t item = 0; item < InventoryItem::InventoryCount; item++) {
            // Validate inventory item index before accessing
            if (item >= InventoryItemNames.size()) {
              throw std::runtime_error("Invalid inventory item index: " + std::to_string(item));
            }

            actor->stats.add(InventoryItemNames[item], "stolen", actor->group_name, agent_target->inventory[item]);
            actor->update_inventory(static_cast<InventoryItem>(item), agent_target->inventory[item]);
            agent_target->update_inventory(static_cast<InventoryItem>(item), -agent_target->inventory[item]);
          }
        }

        return true;
      }
    }

    // Check for objects if no agent was found or armor blocked
    target_loc.layer = GridLayer::Object_Layer;
    GridObject* obj_at_location = safe_object_at(target_loc);

    MettaObject* object_target = nullptr;
    if (obj_at_location != nullptr) {
      object_target = dynamic_cast<MettaObject*>(obj_at_location);
      if (obj_at_location != nullptr && object_target == nullptr) {
        throw std::runtime_error("Object at target location is not a MettaObject");
      }
    }

    if (object_target) {
      actor->stats.incr(_stats.target[object_target->_type_id]);
      actor->stats.incr(_stats.target[object_target->_type_id], actor->group_name);
      object_target->hp -= 1;
      actor->stats.incr("damage", ObjectTypeNames[object_target->_type_id]);
      if (object_target->hp <= 0) {
        _grid->remove_object(object_target);
        actor->stats.incr("destroyed", ObjectTypeNames[object_target->_type_id]);
      }
      return true;
    }

    return false;
  }
};
}  // namespace Actions
#endif  // ATTACK_HPP