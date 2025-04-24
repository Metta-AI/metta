#ifndef NOOP_HPP
#define NOOP_HPP

#include <string>

#include "action_handler.hpp"
#include "objects/agent.hpp"

class Noop : public ActionHandler {
public:
  Noop(const ActionConfig& cfg) : ActionHandler(cfg, "noop") {}

  unsigned char max_arg() const override {
    return 0;
  }

protected:
  bool _handle_action(unsigned int actor_id, Agent* actor, ActionArg arg) override {
    return true;
  }
};

#endif  // NOOP_HPP
