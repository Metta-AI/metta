#ifndef ACTIONS_MOVE_8WAY_OPTIMIZED_HPP_
#define ACTIONS_MOVE_8WAY_OPTIMIZED_HPP_

#include <memory>
#include <string>

#include "action_handler.hpp"
#include "grid_object.hpp"
#include "objects/agent.hpp"
#include "types.hpp"

// Template implementation for compile-time optimization
template<bool NoAgentInterference>
class Move8WayImpl : public ActionHandler {
public:
  explicit Move8WayImpl(const ActionConfig& cfg)
    : ActionHandler(cfg, "move_8way") {}

  unsigned char max_arg() const override {
    return 7;  // 8 directions
  }

protected:
  bool _handle_action(Agent* actor, ActionArg arg) override {
    // 8-way movement: direct movement in 8 directions including diagonals
    GridLocation current_location = actor->location;
    GridLocation target_location = current_location;
    Orientation new_orientation = actor->orientation;

    switch (arg) {
      case 0:  // North
        target_location.r -= 1;
        new_orientation = Orientation::Up;
        break;
      case 1:  // Northeast
        target_location.r -= 1;
        target_location.c += 1;
        new_orientation = Orientation::Up;
        if (!_is_valid_square(target_location)) {
            // Tries clockwise adj cardinal dir (East)
            target_location.r += 1;
            new_orientation = Orientation::Right;
            if (!_is_valid_square(target_location)) {
                // Tries counter-clockwise adj cardinal dir (North)
                target_location.r -= 1;
                target_location.c -= 1;
                new_orientation = Orientation::Up;
                break;
            }
            break;
        }
        break;
      case 2:  // East
        target_location.c += 1;
        new_orientation = Orientation::Right;
        break;
      case 3:  // Southeast
        target_location.r += 1;
        target_location.c += 1;
        new_orientation = Orientation::Down;
        if (!_is_valid_square(target_location)) {
            // Tries clockwise adj cardinal dir (South)
            target_location.c -= 1;
            new_orientation = Orientation::Down;
            if (!_is_valid_square(target_location)) {
                // Tries counter-clockwise adj cardinal dir (East)
                target_location.r -= 1;
                target_location.c += 1;
                new_orientation = Orientation::Right;
                break;
            }
            break;
        }
        break;
      case 4:  // South
        target_location.r += 1;
        new_orientation = Orientation::Down;
        break;
      case 5:  // Southwest
        target_location.r += 1;
        target_location.c -= 1;
        new_orientation = Orientation::Down;
        if (!_is_valid_square(target_location)) {
            // Tries clockwise adj cardinal dir (West)
            target_location.r -= 1;
            new_orientation = Orientation::Left;
            if (!_is_valid_square(target_location)) {
                // Tries counter-clockwise adj cardinal dir (South)
                target_location.r += 1;
                target_location.c += 1;
                new_orientation = Orientation::Down;
                break;
            }
            break;
        }
        break;
      case 6:  // West
        target_location.c -= 1;
        new_orientation = Orientation::Left;
        break;
      case 7:  // Northwest
        target_location.r -= 1;
        target_location.c -= 1;
        new_orientation = Orientation::Up;
        if (!_is_valid_square(target_location)) {
            // Tries clockwise adj cardinal dir (North)
            target_location.c += 1;
            new_orientation = Orientation::Up;
            if (!_is_valid_square(target_location)) {
                // Tries counter-clockwise adj cardinal dir (West)
                target_location.r += 1;
                target_location.c -= 1;
                new_orientation = Orientation::Left;
                break;
            }
            break;
        }
        break;
      default:
        return false;
    }

    // Check if we can move to the target location
    if (!_is_valid_square(target_location)) {
      return false;
    }

    // Update orientation
    actor->orientation = new_orientation;

    // Move the agent using compile-time selected function
    if constexpr (NoAgentInterference) {
      return _grid->ghost_move_object(actor->id, target_location);
    } else {
      return _grid->move_object(actor->id, target_location);
    }
  }

private:
  bool _is_valid_square(GridLocation target_location) const {
    if (!_grid->is_valid_location(target_location)) {
      return false;
    }
    
    // Compile-time branch elimination
    if constexpr (NoAgentInterference) {
      // Ghost mode: only check object layer
      return _grid->is_empty_at_layer(target_location.r, target_location.c, GridLayer::ObjectLayer);
    } else {
      // Normal mode: check all layers
      return _grid->is_empty(target_location.r, target_location.c);
    }
  }
};

// Factory function to create the appropriate Move8Way implementation at runtime
inline std::unique_ptr<ActionHandler> createMove8WayAction(
    const ActionConfig& cfg,
    bool no_agent_interference) {
  
  if (no_agent_interference) {
    return std::make_unique<Move8WayImpl<true>>(cfg);
  } else {
    return std::make_unique<Move8WayImpl<false>>(cfg);
  }
}

#endif  // ACTIONS_MOVE_8WAY_OPTIMIZED_HPP_