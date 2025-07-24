#ifndef ACTIONS_MOVE_HPP_
#define ACTIONS_MOVE_HPP_

#include <string>
// #include <vector>

#include "action_handler.hpp"
#include "grid_object.hpp"
#include "objects/agent.hpp"
#include "types.hpp"

// MOVE has been modified so that moves onto empty squares are the most prioritized moves, and then after all other actions
// are processed, any move that was moving into another agent is processed for each agent in their random order,
// moving them in the chosen direction if nothing is blocking them, and swapping them with the agent there if they are blocked by one

struct DeferredMove {
    GridObjectId agent_id;
    GridLocation target_location;
    ActionArg arg;
};

class Move : public ActionHandler {
public:
    std::vector<DeferredMove> deferred_moves;

    explicit Move(const ActionConfig& cfg) : ActionHandler(cfg, "move") {}

    void clear_deferred_moves() {
        deferred_moves.clear();
    }

    void process_deferred_moves() {
        // Process all deferred moves: swap if target is still an agent
        for (const auto& move : deferred_moves) {
            Agent* agent = dynamic_cast<Agent*>(_grid->object(move.agent_id));
            GridObject* target = _grid->object_at(move.target_location);
            Agent* target_agent = dynamic_cast<Agent*>(target);
            if (agent && target_agent) {
                _grid->swap_objects(agent->id, target_agent->id);
            }
            else if (agent) {
                _grid->move_object(agent->id, move.target_location);
            }
        }
        deferred_moves.clear();
    }

    unsigned char max_arg() const override {
        return 1;  // 0 = move forward, 1 = move backward
    }

protected:
    bool _handle_action(Agent* actor, ActionArg arg) override {
        // Move action: agents move in a direction without changing orientation
        // arg == 0: Move forward in current direction
        // arg == 1: Move backward (reverse direction) while maintaining facing direction

        Orientation move_direction = static_cast<Orientation>(actor->orientation);

        if (arg == 1) {
            move_direction = get_opposite_direction(move_direction);
        }

        GridLocation current_location = actor->location;
        GridLocation target_location = _grid->relative_location(current_location, move_direction);

        // check if we are blocked by an obstacle
        if (!_grid->is_empty(target_location.r, target_location.c)) {
          // If we are attempting to move into an agent, then we defer the move for later instead of rejecting it
          // so that we can allow for agents to pass through each other
            GridObject* target = _grid->object_at(target_location);
            if (target && dynamic_cast<Agent*>(target)) {

                deferred_moves.push_back({actor->id, target_location, arg});
                return true;
            }
            return false;
        }

        return _grid->move_object(actor->id, target_location);
    }

private:
    // Get the opposite direction (for backward movement)
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

#endif  // ACTIONS_MOVE_HPP_
