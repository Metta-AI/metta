#ifndef PACKAGES_METTAGRID_CPP_INCLUDE_METTAGRID_ACTIONS_SWAP_HPP_
#define PACKAGES_METTAGRID_CPP_INCLUDE_METTAGRID_ACTIONS_SWAP_HPP_

#include <string>

#include "actions/action_handler.hpp"
#include "core/grid.hpp"
#include "core/grid_object.hpp"
#include "core/types.hpp"
#include "objects/agent.hpp"

class Swap : public ActionHandler {
public:
  explicit Swap(const ActionConfig& cfg) : ActionHandler(cfg, "swap") {}

  std::vector<Action> create_actions() override {
    return {Action(this, "swap", 0)};
  }

protected:
  bool _handle_action(Agent& actor, ActionArg /*arg*/) override {
    // target the square we are facing
    GridLocation target_loc = _grid->relative_location(actor.location, actor.orientation);

    GridObject* target = this->_grid->object_at(target_loc);
    if (target && target->swappable()) {
      actor.stats.incr("action." + this->_action_name + "." + target->type_name);
      this->_grid->swap_objects(actor, *target);
      return true;
    }

    return false;
  }
};

#endif  // PACKAGES_METTAGRID_CPP_INCLUDE_METTAGRID_ACTIONS_SWAP_HPP_
