#ifndef ACTIONS_ATTACK_HPP_
#define ACTIONS_ATTACK_HPP_

#include <map>
#include <string>
#include <vector>

#include "action_handler.hpp"
#include "grid_object.hpp"
#include "objects/agent.hpp"
#include "objects/constants.hpp"
#include "objects/metta_object.hpp"
#include "types.hpp"

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
    return 9;
  }

protected:
  std::map<InventoryItem, InventoryQuantity> _defense_resources;

  bool _handle_action(Agent* actor, ActionArg arg) override {
    // Scan the 9 squares in front of the agent (3x3 grid)
    int agents_found = 0;
    Agent* target = nullptr;
    GridLocation last_agent_loc;
    short num_skipped = 0;

    // Scan distances 1-3
    for (int distance = 1; distance <= 3; distance++) {
      // Scan offsets -1, 0, 1 (left, center, right)
      for (int offset = -1; offset <= 1; offset++) {
        GridLocation target_loc =
            _grid->relative_location(actor->location, static_cast<Orientation>(actor->orientation), distance, offset);

        GridObject* obj = _grid->object_at(target_loc);
        if (obj == nullptr) {
          continue;
        }

        target = dynamic_cast<Agent*>(obj);
        if (num_skipped == arg) {
          break;
        }
        num_skipped++;
      }
    }
    if (target) {
      return _handle_target(actor, target);
    }
    return false;
  }

  bool _handle_target(Agent* actor, Agent* target) {
    bool was_frozen = false;
    if (target) {
      // Track attack targets
      actor->stats.incr("action." + _action_name + "." + target->type_name);
      actor->stats.incr("action." + _action_name + "." + target->type_name + "." + actor->group_name);
      actor->stats.incr("action." + _action_name + "." + target->type_name + "." + actor->group_name + "." +
                        target->group_name);

      if (target->group_name == actor->group_name) {
        actor->stats.incr("attack.own_team." + actor->group_name);
      } else {
        actor->stats.incr("attack.other_team." + actor->group_name);
      }

      was_frozen = target->frozen > 0;

      bool blocked = _defense_resources.size() > 0;
      for (const auto& [item, amount] : _defense_resources) {
        if (target->inventory[item] < amount) {
          blocked = false;
          break;
        }
      }

      if (blocked) {
        // Consume the defense resources
        for (const auto& [item, amount] : _defense_resources) {
          int used_amount = std::abs(target->update_inventory(item, -amount));
          assert(used_amount == amount);
        }

        actor->stats.incr("attack.blocked." + target->group_name);
        actor->stats.incr("attack.blocked." + target->group_name + "." + actor->group_name);
        return true;
      } else {
        target->frozen = target->freeze_duration;

        if (!was_frozen) {
          // Actor (attacker) stats
          actor->stats.incr("attack.win." + actor->group_name);
          actor->stats.incr("attack.win." + actor->group_name + "." + target->group_name);

          // Target (victim) stats - these should be on agent_target, not actor
          target->stats.incr("attack.loss." + target->group_name);
          target->stats.incr("attack.loss." + target->group_name + "." + actor->group_name);

          if (target->group_name == actor->group_name) {
            actor->stats.incr("attack.win.own_team." + actor->group_name);
            target->stats.incr("attack.loss.from_own_team." + target->group_name);
          } else {
            actor->stats.incr("attack.win.other_team." + actor->group_name);
            target->stats.incr("attack.loss.from_other_team." + target->group_name);
          }

          // Collect all items to steal first, then apply changes, since the changes
          // can delete keys from the agent's inventory.
          std::vector<std::pair<InventoryItem, int>> items_to_steal;
          for (const auto& [item, amount] : target->inventory) {
            items_to_steal.emplace_back(item, amount);
          }

          // Now apply the stealing
          for (const auto& [item, amount] : resources_to_steal) {
            InventoryDelta stolen = actor->update_inventory(item, static_cast<InventoryDelta>(amount));

            target->update_inventory(item, -stolen);
            if (stolen > 0) {
              actor->stats.add(actor->stats.inventory_item_name(item) + ".stolen." + actor->group_name, stolen);
              // Also track what was stolen from the victim's perspective
              target->stats.add(target->stats.inventory_item_name(item) + ".stolen_from." + target->group_name, stolen);
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
