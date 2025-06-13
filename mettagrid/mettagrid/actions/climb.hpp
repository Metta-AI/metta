#ifndef METTAGRID_METTAGRID_ACTIONS_CLIMB_HPP_
#define METTAGRID_METTAGRID_ACTIONS_CLIMB_HPP_

#include <string>

#include "action_handler.hpp"
#include "grid_object.hpp"
#include "objects/agent.hpp"
#include "objects/stairs.hpp"

class Climb : public ActionHandler {
public:
  explicit Climb(const ActionConfig& cfg) : ActionHandler(cfg, "climb") {}

  unsigned char max_arg() const override {
    return 1;  // 0 = up, 1 = down
  }

protected:
  bool _handle_action(Agent* actor, ActionArg arg) override {
    bool going_up = (arg == 0);

    // Check current Z-level
    if (going_up && actor->z_level > 0) {
      return false;  // Already at top level
    }
    if (!going_up && actor->z_level == 0) {
      return false;  // Already at ground level
    }

    // Check if there are stairs at the current location
    GridLocation stairs_loc = actor->location;
    stairs_loc.layer = GridLayer::Object_Layer;

    GridObject* obj = _grid->object_at(stairs_loc);
    if (!obj || obj->_type_id != ObjectType::StairsT) {
      return false;  // No stairs here
    }

    // Change Z-level
    if (going_up) {
      actor->z_level = 1;
    } else {
      actor->z_level = 0;
    }

    return true;
  }
};

#endif  // METTAGRID_METTAGRID_ACTIONS_CLIMB_HPP_
