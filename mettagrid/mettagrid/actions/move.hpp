#ifndef MOVE_HPP
#define MOVE_HPP
#include <cstdint>  // Added for fixed-width integer types
#include <string>

#include "action_handler.hpp"
#include "grid_object.hpp"
#include "objects/agent.hpp"

class Move : public ActionHandler {
public:
  Move(const ActionConfig& cfg) : ActionHandler(cfg, "move") {}

  uint8_t max_arg() const override {
    return 1;
  }

  ActionHandler* clone() const override {
    return new Move(*this);
  }

protected:
  bool _handle_action(uint32_t actor_id, Agent* actor, ActionArg arg) override {
    uint16_t direction = arg;
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

#endif  // MOVE_HPP