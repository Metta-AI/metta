#ifndef ROTATE_HPP
#define ROTATE_HPP

#include <cstdint>  // Added for fixed-width integer types
#include <string>

#include "action_handler.hpp"
#include "objects/agent.hpp"

class Rotate : public ActionHandler {
public:
  Rotate(const ActionConfig& cfg) : ActionHandler(cfg, "rotate") {}

  uint8_t max_arg() const override {
    return 3;
  }

  ActionHandler* clone() const override {
    return new Rotate(*this);
  }

protected:
  bool _handle_action(uint32_t actor_id, Agent* actor, ActionArg arg) override {
    uint16_t orientation = arg;
    actor->orientation = orientation;
    return true;
  }
};

#endif  // ROTATE_HPP