#ifndef METTAGRID_METTAGRID_ACTIONS_MOVE_HPP_
#define METTAGRID_METTAGRID_ACTIONS_MOVE_HPP_

#include <string>

#include "action_handler.hpp"
#include "grid_object.hpp"
#include "objects/agent.hpp"
#include "objects/constants.hpp"

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

    // Check if the new location is walkable based on Z-level
    if (actor->z_level == 0) {
      // Ground level - check for walls and other objects normally
      if (!_grid->is_empty(new_loc.r, new_loc.c)) {
        // Check if there's a TallBridge - we can walk under it
        GridLocation obj_loc(new_loc.r, new_loc.c, GridLayer::Object_Layer);
        GridObject* obj = _grid->object_at(obj_loc);
        if (!obj || obj->_type_id != ObjectType::TallBridgeT) {
          return false;  // Blocked by something that's not a TallBridge
        }
      }
    } else {
      // Upper Z-level - walls act as floors, only check agent layer
      GridLocation agent_loc(new_loc.r, new_loc.c, GridLayer::Agent_Layer);
      if (_grid->object_at(agent_loc) != nullptr) {
        // Check if the other agent is at the same Z-level
        Agent* other_agent = static_cast<Agent*>(_grid->object_at(agent_loc));
        if (other_agent && other_agent->z_level == actor->z_level) {
          return false;  // Blocked by agent at same Z-level
        }
      }

      // Check if there's a wall below us (acts as floor) or a TallBridge
      GridLocation obj_loc(new_loc.r, new_loc.c, GridLayer::Object_Layer);
      GridObject* obj = _grid->object_at(obj_loc);
      if (!obj || (obj->_type_id != ObjectType::WallT && obj->_type_id != ObjectType::TallBridgeT)) {
        return false;  // No floor to walk on at upper level
      }
    }

    return _grid->move_object(actor->id, new_loc);
  }
};

#endif  // METTAGRID_METTAGRID_ACTIONS_MOVE_HPP_
