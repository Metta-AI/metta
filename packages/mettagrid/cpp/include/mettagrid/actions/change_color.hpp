#ifndef PACKAGES_METTAGRID_CPP_INCLUDE_METTAGRID_ACTIONS_CHANGE_COLOR_HPP_
#define PACKAGES_METTAGRID_CPP_INCLUDE_METTAGRID_ACTIONS_CHANGE_COLOR_HPP_

#include <string>

#include "actions/action_handler.hpp"
#include "core/types.hpp"
#include "objects/agent.hpp"

class ChangeColor : public ActionHandler {
public:
  explicit ChangeColor(const ActionConfig& cfg) : ActionHandler(cfg, "change_color") {}

  unsigned char max_arg() const override {
    return 3;  // support fine and coarse adjustment
  }

protected:
  bool _handle_action(Agent& actor, ActionArg arg) override {
    // Note: 'color' is uint8_t which naturally wraps at 256.
    // This could be interpreted as circular hue behavior (red -> orange -> ... -> violet -> red)

    // Calculate step size once (integer division is intentional)
    const uint8_t step_size = static_cast<uint8_t>(255 / (max_arg() + 1));

    if (arg == 0) {  // Increment
      actor.color = static_cast<uint8_t>(actor.color + 1);
    } else if (arg == 1) {  // Decrement
      actor.color = static_cast<uint8_t>(actor.color - 1);
    } else if (arg == 2) {  // Large increment
      actor.color = static_cast<uint8_t>(actor.color + step_size);
    } else if (arg == 3) {  // Large decrement
      actor.color = static_cast<uint8_t>(actor.color - step_size);
    }

    return true;
  }
};

#endif  // PACKAGES_METTAGRID_CPP_INCLUDE_METTAGRID_ACTIONS_CHANGE_COLOR_HPP_
