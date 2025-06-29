#ifndef METTAGRID_METTAGRID_ACTIONS_ATTACK_NEAREST_HPP_
#define METTAGRID_METTAGRID_ACTIONS_ATTACK_NEAREST_HPP_

#include <string>

#include "attack.hpp"
#include "grid_object.hpp"
#include "objects/agent.hpp"
#include "objects/constants.hpp"

class AttackNearest : public Attack {
public:
  explicit AttackNearest(const ActionConfig& cfg) : Attack(cfg, "attack_nearest") {}

  unsigned char max_arg() const override {
    return 0;
  }

protected:
  bool _handle_action(Agent* actor, ActionArg arg) override {
    if (actor->update_inventory(InventoryItem::laser, -1) == 0) {
      return false;
    }

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
        Agent* agent_target = dynamic_cast<Agent*>(_grid->object_at(target_loc));
        if (agent_target) {
          return _handle_target(actor, target_loc);
        }
      }
    }

    return false;
  }
};

#endif  // METTAGRID_METTAGRID_ACTIONS_ATTACK_NEAREST_HPP_
