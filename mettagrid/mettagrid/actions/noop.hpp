#ifndef NOOP_HPP
#define NOOP_HPP

#include <cstdint>  // Added for fixed-width integer types
#include <string>

#include "action_handler.hpp"
#include "objects/agent.hpp"

class Noop : public ActionHandler {
public:
  Noop(const ActionConfig& cfg) : ActionHandler(cfg, "noop") {}

  uint8_t max_arg() const override {
    return 0;
  }

  ActionHandler* clone() const override {
    return new Noop(*this);
  }

protected:
  bool _handle_action(uint32_t actor_id, Agent* actor, ActionArg arg) override {
    return true;
  }
};

#endif  // NOOP_HPP