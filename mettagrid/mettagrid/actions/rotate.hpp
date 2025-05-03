#ifndef ROTATE_HPP
#define ROTATE_HPP

#include <string>

#include "action_handler.hpp"
#include "objects/agent.hpp"

class Rotate : public ActionHandler {
public:
  Rotate(const ActionConfig& cfg) : ActionHandler(cfg, "rotate") {}

  unsigned char max_arg() const override {
    return 3;
  }

protected:
  bool _handle_action(unsigned int actor_id, Agent* actor, ActionArg arg) override {
    unsigned short orientation = arg;
    actor->orientation = orientation;
    return true;
  }
};

#endif  // ROTATE_HPP
