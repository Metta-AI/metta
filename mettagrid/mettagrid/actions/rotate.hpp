#ifndef ROTATE_HPP
#define ROTATE_HPP

#include <cstdint>
#include <string>

#include "actions/action_handler.hpp"
#include "objects/agent.hpp"
namespace Actions {
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
  bool _handle_action(uint32_t actor_id, Agent* actor, ActionsType arg) override {
    // Null checks for actor and grid are now handled in the base class

    // Validate that the orientation is within bounds
    if (arg > 3) {
      throw std::runtime_error("Invalid orientation value: " + std::to_string(arg) + " (must be between 0 and 3)");
    }

    // Set the new orientation
    actor->orientation = arg;

    return true;
  }
};
}  // namespace Actions
#endif  // ROTATE_HPP