#ifndef ACTIONS_ATTACK_NEAREST_HPP_
#define ACTIONS_ATTACK_NEAREST_HPP_

#include <string>

#include "attack.hpp"
#include "grid_object.hpp"
#include "objects/agent.hpp"
#include "objects/constants.hpp"

class AttackNearest : public Attack {
public:
  explicit AttackNearest(const AttackActionConfig& cfg) : Attack(cfg, "attack_nearest") {}

  unsigned char max_arg() const override {
    return 0;
  }

protected:
  bool _handle_action(Agent* actor, ActionArg arg) override {
    // Defines the search pattern: check directly in front first, then to one side, then the other.
    const int offsets[] = {0, 1, -1};

    for (int distance = 1; distance < 4; distance++) {
      for (const int offset : offsets) {
        GridLocation target_loc =
            _grid->relative_location(actor->location, static_cast<Orientation>(actor->orientation), distance, offset);

        target_loc.layer = GridLayer::AgentLayer;

        GridObject* obj = _grid->object_at(target_loc);
        if (!obj) continue;

        // we know obj is an Agent because it is in the agent layer
        return _handle_target(static_cast<Agent*>(obj), target_loc);
      }
    }

    return false;
  }
};

#endif  // ACTIONS_ATTACK_NEAREST_HPP_
