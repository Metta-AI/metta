#ifndef ACTIONS_MOVE_OPTIMIZED_HPP_
#define ACTIONS_MOVE_OPTIMIZED_HPP_

#include <memory>
#include <string>

#include "action_handler.hpp"
#include "grid_object.hpp"
#include "objects/agent.hpp"
#include "objects/constants.hpp"
#include "types.hpp"

// Template implementation for compile-time optimization
template<bool NoAgentInterference>
class MoveImpl : public ActionHandler {
public:
  explicit MoveImpl(const ActionConfig& cfg, bool track_movement_metrics = false)
    : ActionHandler(cfg, "move"), _track_movement_metrics(track_movement_metrics) {}

  unsigned char max_arg() const override {
    return 1;  // 0 = move forward, 1 = move backward
  }

protected:
  bool _handle_action(Agent* actor, ActionArg arg) override {
    // Move action: agents move in a direction without changing orientation
    Orientation move_direction = static_cast<Orientation>(actor->orientation);

    if (arg == 1) {
      move_direction = get_opposite_direction(move_direction);
    }

    GridLocation current_location = actor->location;
    GridLocation target_location = _grid->relative_location(current_location, move_direction);

    // Compile-time branch elimination using if constexpr
    bool success = false;
    if constexpr (NoAgentInterference) {
      // Ghost mode: only check object layer, agents can pass through each other
      if (!_grid->is_empty_at_layer(target_location.r, target_location.c, GridLayer::ObjectLayer)) {
        return false;
      }
      success = _grid->ghost_move_object(actor->id, target_location);
    } else {
      // Normal mode: check all layers
      if (!_grid->is_empty(target_location.r, target_location.c)) {
        return false;
      }
      success = _grid->move_object(actor->id, target_location);
    }

    if (success) {
      // Increment visitation count for the new position
      actor->increment_visitation_count(target_location.r, target_location.c);

      // Track movement direction (only if tracking enabled)
      if (_track_movement_metrics) {
        actor->stats.add(std::string("movement.direction.") + OrientationNames[static_cast<int>(move_direction)], 1);
      }
    }

    return success;
  }

private:
  bool _track_movement_metrics;

  static Orientation get_opposite_direction(Orientation orientation) {
    switch (orientation) {
      case Orientation::Up:
        return Orientation::Down;
      case Orientation::Down:
        return Orientation::Up;
      case Orientation::Left:
        return Orientation::Right;
      case Orientation::Right:
        return Orientation::Left;
      default:
        assert(false && "Invalid orientation passed to get_opposite_direction()");
    }
  }
};

// Factory function to create the appropriate Move implementation at runtime
inline std::unique_ptr<ActionHandler> createMoveAction(
    const ActionConfig& cfg, 
    bool track_movement_metrics,
    bool no_agent_interference) {
  
  if (no_agent_interference) {
    return std::make_unique<MoveImpl<true>>(cfg, track_movement_metrics);
  } else {
    return std::make_unique<MoveImpl<false>>(cfg, track_movement_metrics);
  }
}

#endif  // ACTIONS_MOVE_OPTIMIZED_HPP_