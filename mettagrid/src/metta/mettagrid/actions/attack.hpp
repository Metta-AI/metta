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
  std::map<InventoryItem, InventoryQuantity> defense_resources;

  AttackActionConfig(bool enabled,
                     const std::map<InventoryItem, InventoryQuantity>& required_resources,
                     const std::map<InventoryItem, InventoryQuantity>& consumed_resources,
                     const std::map<InventoryItem, InventoryQuantity>& defense_resources)
      : ActionConfig(enabled, required_resources, consumed_resources), defense_resources(defense_resources) {}
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
    if (arg > 9 || arg < 1) {
      return false;
    }

    // Attack positions form a 3x3 grid in front of the agent
    // Visual representation (agent facing up):
    //   7  8  9    (3 cells forward)
    //   4  5  6    (2 cells forward)
    //   1  2  3    (1 cell forward)
    //      A       (Agent position)
    static constexpr std::pair<short, short> ATTACK_POSITIONS[9] = {
        {1, -1},  // arg 1: 1 forward, 1 left
        {1, 0},   // arg 2: 1 forward, straight ahead
        {1, 1},   // arg 3: 1 forward, 1 right
        {2, -1},  // arg 4: 2 forward, 1 left
        {2, 0},   // arg 5: 2 forward, straight ahead
        {2, 1},   // arg 6: 2 forward, 1 right
        {3, -1},  // arg 7: 3 forward, 1 left
        {3, 0},   // arg 8: 3 forward, straight ahead
        {3, 1},   // arg 9: 3 forward, 1 right
    };

    auto [distance, offset] = ATTACK_POSITIONS[arg - 1];
    GridLocation target_loc = _grid->relative_location(actor->location, actor->orientation, distance, offset);

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

      was_frozen = agent_target->frozen > 0;

      bool blocked = _defense_resources.size() > 0;
      for (const auto& [item, amount] : _defense_resources) {
        if (agent_target->inventory[item] < amount) {
          blocked = false;
          break;
        }
      }

      if (blocked) {
        // Consume the defense resources
        for (const auto& [item, amount] : _defense_resources) {
          InventoryDelta used_amount =
              std::abs(agent_target->update_inventory(item, -static_cast<InventoryDelta>(amount)));
          assert(used_amount == static_cast<InventoryDelta>(amount));
        }

        actor->stats.incr("attack.blocked." + agent_target->group_name);
        actor->stats.incr("attack.blocked." + agent_target->group_name + "." + actor->group_name);
        return true;
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

          // Collect all items to steal first, then apply changes, since the changes
          // can delete keys from the agent's inventory.
          std::vector<std::pair<InventoryItem, InventoryQuantity>> resources_to_steal;
          for (const auto& [item, amount] : agent_target->inventory) {
            resources_to_steal.emplace_back(item, amount);
          }

          // Now apply the stealing
          for (const auto& [item, amount] : resources_to_steal) {
            InventoryDelta stolen = actor->update_inventory(item, static_cast<InventoryDelta>(amount));

            agent_target->update_inventory(item, -stolen);
            if (stolen > 0) {
              actor->stats.add(actor->stats.inventory_item_name(item) + ".stolen." + actor->group_name, stolen);
              // Also track what was stolen from the victim's perspective
              agent_target->stats.add(
                  agent_target->stats.inventory_item_name(item) + ".stolen_from." + agent_target->group_name, stolen);
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
