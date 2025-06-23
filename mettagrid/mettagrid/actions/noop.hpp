#ifndef METTAGRID_METTAGRID_ACTIONS_NOOP_HPP_
#define METTAGRID_METTAGRID_ACTIONS_NOOP_HPP_

#include <string>

#include "action_handler.hpp"
#include "objects/agent.hpp"
#include "types.hpp"

class Noop : public ActionHandler {
public:
  explicit Noop(const ActionConfig& cfg) : ActionHandler(cfg, "noop") {}

  unsigned char max_arg() const override {
    return 0;
  }

protected:
  bool _handle_action(Agent* actor, ActionArg arg) override {
    return true;
  }
};

#endif  // METTAGRID_METTAGRID_ACTIONS_NOOP_HPP_
