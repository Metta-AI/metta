#ifndef METTAGRID_METTAGRID_ACTIONS_ROTATE_HPP_
#define METTAGRID_METTAGRID_ACTIONS_ROTATE_HPP_

#include <string>

#include "action_handler.hpp"
#include "objects/agent.hpp"
#include "types.hpp"

class Rotate : public ActionHandler {
public:
  explicit Rotate(const ActionConfig& cfg) : ActionHandler(cfg, "rotate") {}

  unsigned char max_arg() const override {
    return 3;
  }

protected:
  bool _handle_action(Agent* actor, ActionArg arg) override {
    unsigned short orientation = arg;
    actor->orientation = orientation;
    return true;
  }
};

#endif  // METTAGRID_METTAGRID_ACTIONS_ROTATE_HPP_
