#ifndef ACTIONS_MOVE_HPP_
#define ACTIONS_MOVE_HPP_

#include <string>

#include "action_handler.hpp"
#include "grid_object.hpp"
#include "objects/agent.hpp"
#include "types.hpp"

class Move : public ActionHandler {
public:
  explicit Move(const ActionConfig& cfg) : ActionHandler(cfg, "move") {}

  unsigned char max_arg() const override {
    return 1;
  }

protected:
  bool _handle_action(Agent* actor, ActionArg arg) override {
    unsigned short direction = arg;

    Orientation orientation = static_cast<Orientation>(actor->orientation);
    if (direction == 1) {
      if (orientation == Orientation::Up) {
        orientation = Orientation::Down;
      } else if (orientation == Orientation::Down) {
        orientation = Orientation::Up;
      } else if (orientation == Orientation::Left) {
        orientation = Orientation::Right;
      } else if (orientation == Orientation::Right) {
        orientation = Orientation::Left;
      }
    }

    GridLocation old_loc = actor->location;
    GridLocation new_loc = _grid->relative_location(old_loc, orientation);
    if (!_grid->is_empty(new_loc.r, new_loc.c)) {
      return false;
    }
    return _grid->move_object(actor->id, new_loc);
  }
};

#endif  // ACTIONS_MOVE_HPP_
