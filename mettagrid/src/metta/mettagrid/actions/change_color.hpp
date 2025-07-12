#ifndef ACTIONS_CHANGE_COLOR_HPP_
#define ACTIONS_CHANGE_COLOR_HPP_
#include <string>

#include "action_handler.hpp"
#include "objects/agent.hpp"
#include "types.hpp"

class ChangeColor : public ActionHandler {
public:
  explicit ChangeColor(const ActionConfig& cfg) : ActionHandler(cfg, "change_color") {}

  unsigned char max_arg() const override {
    return 3;  // support fine and coarse adjustment
  }

protected:
  bool _handle_action(Agent* actor, ActionArg arg) override {
    // Note: 'color' is interpreted as HSV hue (0-255 range, wrapping)
    if (arg == 0) {  // Increment
      actor->color = (actor->color + 1) % 256;
    } else if (arg == 1) {                        // Decrement
      actor->color = (actor->color + 255) % 256;  // Equivalent to -1 with wrapping
    } else if (arg == 2) {                        // Large increment
      actor->color = (actor->color + 255 / (max_arg() + 1)) % 256;
    } else if (arg == 3) {  // Large decrement
      actor->color = (actor->color + 256 - (255 / (max_arg() + 1))) % 256;
    }
    return true;
  }
};

#endif  // ACTIONS_CHANGE_COLOR_HPP_
