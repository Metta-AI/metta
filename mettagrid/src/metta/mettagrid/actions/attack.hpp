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
    // Force the lookup to the agent layer (ignores non-agent targets)
    target_loc.layer = GridLayer::AgentLayer;

    GridObject* obj = _grid->object_at(target_loc);
    if (!obj) return false;

    Agent* target = static_cast<Agent*>(obj);

    bool was_frozen = target->frozen > 0;

    // === Stat tracking ===
    const std::string& action = _action_name;
    const std::string& actor_group = actor->group_name;
    const std::string& target_group = target->group_name;
    const std::string& target_type = target->type_name;
    bool same_team = (actor_group == target_group);

    std::string base = "action." + action + "." + target_type;
    actor->stats.incr(base);
    actor->stats.incr(base + "." + actor_group);
    actor->stats.incr(base + "." + actor_group + "." + target_group);
    actor->stats.incr(same_team ? "attack.own_team." + actor_group : "attack.other_team." + actor_group);

    if (!_defense_resources.empty()) {
      // it is possible to block this attack if the target has the right inventory
      bool target_can_defend = true;

      for (const auto& [item, amount] : _defense_resources) {
        auto it = target->inventory.find(item);
        if (it == target->inventory.end() || it->second < amount) {
          target_can_defend = false;  // insufficient resources
          break;
        }
      }

      if (target_can_defend) {
        // consume the required defense resources
        for (const auto& [item, amount] : _defense_resources) {
          InventoryDelta used = std::abs(target->update_inventory(item, -static_cast<InventoryDelta>(amount)));
          assert(used == static_cast<InventoryDelta>(amount));
        }

        actor->stats.incr("attack.blocked." + target_group);
        actor->stats.incr("attack.blocked." + target_group + "." + actor_group);
        return true;
      }
    }

    // === Attack succeeds ===
    target->frozen = target->freeze_duration;

    if (!was_frozen) {
      actor->stats.incr("attack.win." + actor_group);
      actor->stats.incr("attack.win." + actor_group + "." + target_group);

      target->stats.incr("attack.loss." + target_group);
      target->stats.incr("attack.loss." + target_group + "." + actor_group);

      actor->stats.incr(same_team ? "attack.win.own_team." + actor_group : "attack.win.other_team." + actor_group);
      target->stats.incr(same_team ? "attack.loss.from_own_team." + target_group
                                   : "attack.loss.from_other_team." + target_group);

      // === Resource stealing ===
      std::vector<std::pair<InventoryItem, InventoryQuantity>> snapshot;
      for (const auto& [item, amount] : target->inventory) snapshot.emplace_back(item, amount);

      for (const auto& [item, amount] : snapshot) {
        InventoryDelta stolen = actor->update_inventory(item, static_cast<InventoryDelta>(amount));
        target->update_inventory(item, -stolen);

        if (stolen > 0) {
          actor->stats.add(actor->stats.inventory_item_name(item) + ".stolen." + actor_group, stolen);
          target->stats.add(target->stats.inventory_item_name(item) + ".stolen_from." + target_group, stolen);
        }
      }
    }

    return true;
  }
};

#endif  // ACTIONS_ATTACK_HPP_
