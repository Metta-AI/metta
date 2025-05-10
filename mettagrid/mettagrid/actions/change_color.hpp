#ifndef CHANGE_COLOR_HPP
#define CHANGE_COLOR_HPP

#include <cstdint>
#include <string>

#include "actions/action_handler.hpp"
#include "objects/agent.hpp"
namespace Actions {
class ChangeColor : public ActionHandler {
public:
  ChangeColor(const ActionConfig& cfg) : ActionHandler(cfg, "change_color") {}

  uint8_t max_arg() const override {
    return 3;
  }

  ActionHandler* clone() const override {
    return new ChangeColor(*this);
  }

protected:
  bool _handle_action(uint32_t actor_id, Agent* actor, ActionsType arg) override {
    // Validate arg (though this check is redundant since we already check in step)
    if (arg > 3) {
      return false;  // Invalid arg is a normal gameplay situation
    }

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
      // Note: Integer division will truncate, which is intentional
    }

    return true;
  }
};
}  // namespace Actions
#endif  // CHANGE_COLOR_HPP