#ifndef ACTIONS_CHANGE_COLOR_HPP_
#define ACTIONS_CHANGE_COLOR_HPP_

#include <string>

#include "action_handler.hpp"
#include "objects/agent.hpp"
#include "types.hpp"

class ChangeColorAction : public ActionHandler {
public:
  explicit ChangeColorAction(const ActionConfig& cfg) : ActionHandler(cfg, "change_color") {
    priority = 2;  // higher than attack
  }

  unsigned char max_arg() const override {
    return 3;  // 4 possible actions (0-3)
  }

protected:
  bool _handle_action(Agent* actor, ActionArg arg) override {
    // Map arg (0-3) to full color range (0-255)
    // arg 0 -> 0, arg 1 -> 85, arg 2 -> 170, arg 3 -> 255
    actor->color = (arg * 255) / max_arg();

    return true;
  }
};

#endif  // ACTIONS_CHANGE_COLOR_HPP_
