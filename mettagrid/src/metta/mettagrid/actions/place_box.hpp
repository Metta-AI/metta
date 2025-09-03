#ifndef ACTIONS_PLACE_BOX_HPP_
#define ACTIONS_PLACE_BOX_HPP_

#include <iostream>
#include <string>

#include "action_handler.hpp"
#include "grid.hpp"
#include "grid_object.hpp"
#include "objects/agent.hpp"
#include "objects/box.hpp"
#include "objects/converter.hpp"
#include "types.hpp"

class PlaceBox : public ActionHandler {
public:
  explicit PlaceBox(const ActionConfig& cfg) : ActionHandler(cfg, "place_box") {}

  unsigned char max_arg() const override {
    return 0;
  }

protected:
  bool _handle_action(Agent* actor, ActionArg /*arg*/) override {
    GridLocation target_loc = _grid->relative_location(actor->location, static_cast<Orientation>(actor->orientation));
    target_loc.layer = GridLayer::ObjectLayer;

    if (!_grid->is_empty(target_loc.r, target_loc.c)) {
      return false;
    }

    if (actor->box) {
      _grid->move_object(actor->box->id, target_loc);
    }

    actor->stats.add("box.created", 1.0f);
    return true;
  }
};

#endif  // ACTIONS_PLACE_BOX_HPP_
