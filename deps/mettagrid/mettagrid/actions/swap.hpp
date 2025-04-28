#ifndef SWAP_HPP
#define SWAP_HPP

#include <string>

#include "action_handler.hpp"
#include "grid.hpp"
#include "grid_object.hpp"
#include "objects/agent.hpp"

class Swap : public ActionHandler {
public:
  Swap(const ActionConfig& cfg) : ActionHandler(cfg, "swap") {}

  unsigned char max_arg() const override {
    return 0;
  }

protected:
  bool _handle_action(unsigned int actor_id, Agent* actor, ActionArg arg) override {
    GridLocation target_loc = _grid->relative_location(actor->location, static_cast<Orientation>(actor->orientation));
    MettaObject* target = static_cast<MettaObject*>(_grid->object_at(target_loc));
    if (target == nullptr) {
      target_loc.layer = GridLayer::Object_Layer;
      target = static_cast<MettaObject*>(_grid->object_at(target_loc));
    }
    if (target == nullptr) {
      return false;
    }

    if (!target->swappable()) {
      return false;
    }

    actor->stats.incr("swap", _stats.target[target->_type_id]);

    _grid->swap_objects(actor->id, target->id);
    return true;
  }
};

#endif  // SWAP_HPP
