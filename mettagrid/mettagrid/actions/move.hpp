#ifndef MOVE_HPP
#define MOVE_HPP
#include <cstdint>
#include <string>

#include "actions/action_handler.hpp"
#include "grid_object.hpp"
#include "objects/agent.hpp"
namespace Actions {
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
  bool _handle_action(uint32_t actor_id, Agent* actor, c_actions_type arg) override {
    // Null checks for actor and grid are now handled in the base class

    // Get the direction from the action argument
    uint16_t direction = arg;

    // Use the base class utility method for orientation validation
    validate_orientation(actor);

    // Get the current orientation and possibly reverse it
    Orientation orientation = static_cast<Orientation>(actor->orientation);
    if (direction == 1) {
      // Reverse the orientation
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

    // Store the old location
    GridLocation old_loc = actor->location;

    // Calculate the new location
    GridLocation new_loc = _grid->relative_location(old_loc, orientation);

    // Use the base class utility method to check if location is valid
    if (!is_valid_location(new_loc)) {
      return false;  // Moving out of bounds - not an error
    }

    // Check if the target location is empty
    if (!_grid->is_empty(new_loc.r, new_loc.c)) {
      return false;  // Space occupied - not an error
    }

    // Try to move the object and return the result
    bool move_result = _grid->move_object(actor->id, new_loc);

    return move_result;
  }
};
}  // namespace Actions
#endif  // MOVE_HPP