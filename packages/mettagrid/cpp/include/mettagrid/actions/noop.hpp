#ifndef PACKAGES_METTAGRID_CPP_INCLUDE_METTAGRID_ACTIONS_NOOP_HPP_
#define PACKAGES_METTAGRID_CPP_INCLUDE_METTAGRID_ACTIONS_NOOP_HPP_

#include <string>

#include "actions/action_handler.hpp"
#include "core/types.hpp"
#include "objects/agent.hpp"

class Noop : public ActionHandler {
public:
  explicit Noop(const ActionConfig& cfg) : ActionHandler(cfg, "noop") {}

  unsigned char max_arg() const override {
    return 0;
  }

protected:
  bool _handle_action(Agent& /*actor*/, ActionArg /*arg*/) override {
    return true;
  }
};

#endif  // PACKAGES_METTAGRID_CPP_INCLUDE_METTAGRID_ACTIONS_NOOP_HPP_
