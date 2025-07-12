#ifndef ACTIONS_CHANGE_COLOR_HPP_
#define ACTIONS_CHANGE_COLOR_HPP_

#include <string>

#include "action_handler.hpp"
#include "objects/agent.hpp"
#include "types.hpp"

// Color change action for multi-agent communication
// Uses direct mapping to make communication instantaneous
// Example: 4 colors map to quadrants of color wheel (red/green/cyan/purple)
//          8 colors add intermediate hues while preserving original positions
// This is inserting a bias that our communication channel is a continuum variable
// see discussion in PR #1435
// TODO: drop "color" and change this action to "signal"
class ChangeColorAction : public ActionHandler {
public:
  explicit ChangeColorAction(const ActionConfig& cfg) : ActionHandler(cfg, "change_color") {}

  unsigned char max_arg() const override {
    return 3;  // 4 possible actions (0-3)
  }

protected:
  bool _handle_action(Agent* actor, ActionArg arg) override {
    // Map action argument to evenly distributed values [0, 255)
    // For n colors: 0, 255/n, 2*255/n, ..., (n-1)*255/n
    actor->color = (arg * 255) / max(max_arg() + 1, 1);

    return true;
  }
};

#endif  // ACTIONS_CHANGE_COLOR_HPP_
