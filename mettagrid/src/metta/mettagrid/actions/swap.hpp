#ifndef ACTIONS_SWAP_HPP_
#define ACTIONS_SWAP_HPP_

#include <string>

#include "action_handler.hpp"
#include "grid.hpp"
#include "grid_object.hpp"
#include "objects/agent.hpp"
#include "types.hpp"

class Swap : public ActionHandler {
public:
  explicit Swap(const ActionConfig& cfg) : ActionHandler(cfg, "swap") {}

  unsigned char max_arg() const override {
    return 0;
  }

protected:
  bool _handle_action(Agent* actor, ActionArg /*arg*/) override {
    // target the square we are facing
    GridLocation target_loc = _grid->relative_location(actor->location, static_cast<Orientation>(actor->orientation));

    // Get all objects at target location, indexed by layer
    auto objects = this->_grid->objects_at(target_loc.r, target_loc.c);

    // Check layers in swap priority order
    const auto layers = {GridLayer::ObjectLayer, GridLayer::AgentLayer};

    for (auto layer : layers) {
      GridObject* target = objects[layer];
      if (target && target->swappable()) {
        actor->stats.incr("action." + this->_action_name + "." + target->type_name);
        this->_grid->swap_objects(actor->id, target->id);
        return true;
      }
    }

    return false;
  }
};

#endif  // ACTIONS_SWAP_HPP_
