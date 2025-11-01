#ifndef PACKAGES_METTAGRID_CPP_INCLUDE_METTAGRID_ACTIONS_NOOP_HPP_
#define PACKAGES_METTAGRID_CPP_INCLUDE_METTAGRID_ACTIONS_NOOP_HPP_

#include <string>

#include "actions/action_handler.hpp"
#include "core/types.hpp"
#include "objects/agent.hpp"

class Noop : public ActionHandler {
public:
  explicit Noop(const ActionConfig& cfg) : ActionHandler(cfg, "noop") {}

  std::vector<Action> create_actions() override {
    std::vector<Action> actions;
    actions.emplace_back(this, "noop", 0);
    return actions;
  }

protected:
  bool _handle_action(Agent& /*actor*/, ActionArg /*arg*/) override {
    return true;
  }
};

#endif  // PACKAGES_METTAGRID_CPP_INCLUDE_METTAGRID_ACTIONS_NOOP_HPP_
