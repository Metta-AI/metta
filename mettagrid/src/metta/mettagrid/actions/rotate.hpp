#ifndef ACTIONS_ROTATE_HPP_
#define ACTIONS_ROTATE_HPP_

#include <string>

#include "action_handler.hpp"
#include "objects/agent.hpp"
#include "types.hpp"

class Rotate : public ActionHandler {
public:
  explicit Rotate(const ActionConfig& cfg, bool track_movement_metrics = false)
      : ActionHandler(cfg, "rotate"), _track_movement_metrics(track_movement_metrics) {}

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
      static const char* direction_names[] = {"up", "down", "left", "right"};
      actor->stats.add(std::string("movement.rotation.to_") + direction_names[static_cast<int>(orientation)], 1);
    }

    return true;
  }

private:
  bool _track_movement_metrics;
};

#endif  // ACTIONS_ROTATE_HPP_
