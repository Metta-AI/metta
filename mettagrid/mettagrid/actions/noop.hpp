#ifndef NOOP_HPP
#define NOOP_HPP

#include <cstdint>
#include <string>

#include "actions/action_handler.hpp"
#include "objects/agent.hpp"
namespace Actions {
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
  bool _handle_action(uint32_t actor_id, Agent* actor, ActionsType arg) override {
    // Null checks for actor and grid are already handled in the base class

    // Noop always succeeds
    return true;
  }
};
}  // namespace Actions
#endif  // NOOP_HPP