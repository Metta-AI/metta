#ifndef ACTIONS_ROTATE_HPP_
#define ACTIONS_ROTATE_HPP_

#include <string>

#include "action_handler.hpp"
#include "objects/agent.hpp"
#include "types.hpp"

class Rotate : public ActionHandler {
public:
  explicit Rotate(const ActionConfig& cfg) : ActionHandler(cfg, "rotate") {}

  unsigned char max_arg() const override {
    return 3;
  }

protected:
  bool _handle_action(Agent* actor, ActionArg arg) override {
    // Orientation: Up = 0, Down = 1, Left = 2, Right = 3
    Orientation orientation = static_cast<Orientation>(arg);
    actor->orientation = orientation;

    // Track which orientation the agent rotated to (only if tracking enabled)
    if (_track_movement_metrics) {
      actor->movement_counters.rotations[static_cast<int>(orientation)]++;
    }

    return true;
  }
};

#endif  // ACTIONS_ROTATE_HPP_
