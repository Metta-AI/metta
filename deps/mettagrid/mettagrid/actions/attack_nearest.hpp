#ifndef ATTACK_NEAREST_HPP
#define ATTACK_NEAREST_HPP

#include <string>

#include "attack.hpp"
#include "grid_object.hpp"
#include "objects/agent.hpp"
#include "objects/constants.hpp"

class AttackNearest : public Attack {
public:
  AttackNearest(const ActionConfig& cfg) : Attack(cfg, "attack_nearest") {}

  unsigned char max_arg() const override {
    return 0;
  }

protected:
  bool _handle_action(unsigned int actor_id, Agent* actor, ActionArg arg) override {
    if (actor->inventory[InventoryItem::laser] == 0) {
      return false;
    }

    actor->update_inventory(InventoryItem::laser, -1);

    // Scan the space to find the nearest agent. Prefer the middle (offset 0) before the edges (offset -1, 1).
    for (int distance = 1; distance < 4; distance++) {
      for (int i = 0; i < 3; i++) {
        int offset = i;
        if (offset == 2) {
          // Sort of a mod 3 operation.
          offset = -1;
        }
        GridLocation target_loc =
            _grid->relative_location(actor->location, static_cast<Orientation>(actor->orientation), distance, offset);

        target_loc.layer = GridLayer::Agent_Layer;
        Agent* agent_target = static_cast<Agent*>(_grid->object_at(target_loc));
        if (agent_target) {
          return _handle_target(actor_id, actor, target_loc);
        }
      }
    }

    return false;
  }
};

#endif  // ATTACK_NEAREST_HPP
