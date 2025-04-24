#ifndef CHANGE_COLOR_HPP
#define CHANGE_COLOR_HPP

#include <string>

#include "action_handler.hpp"
#include "objects/agent.hpp"

class ChangeColorAction : public ActionHandler {
public:
  ChangeColorAction(const ActionConfig& cfg) : ActionHandler(cfg, "change_color") {}

  unsigned char max_arg() const override {
    return 3;
  }

protected:
  bool _handle_action(unsigned int actor_id, Agent* actor, ActionArg arg) override {
    if (arg == 0) {  // Increment
      if (actor->color < 255) {
        actor->color += 1;
      }
    } else if (arg == 1) {  // Decrement
      if (actor->color > 0) {
        actor->color -= 1;
      }
    } else if (arg == 2) {  // Double
      if (actor->color <= 127) {
        actor->color *= 2;
      }
    } else if (arg == 3) {  // Half
      actor->color = actor->color / 2;
    }

    return true;
  }
};

#endif  // CHANGE_COLOR_HPP
