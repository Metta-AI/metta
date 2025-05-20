#ifndef SWAP_HPP
#define SWAP_HPP

#include <cstdint>
#include <string>

#include "actions/action_handler.hpp"
#include "grid.hpp"
#include "grid_object.hpp"
#include "objects/agent.hpp"
namespace Actions {
class Swap : public ActionHandler {
public:
  Swap(const ActionConfig& cfg) : ActionHandler(cfg, "swap") {}

  uint8_t max_arg() const override {
    return 0;
  }

  ActionHandler* clone() const override {
    return new Swap(*this);
  }

protected:
  bool _handle_action(uint32_t actor_id, Agent* actor, c_actions_type arg) override {
    // Null checks for actor and grid are now handled in the base class

    // Validate orientation
    validate_orientation(actor);

    // Get target location
    GridLocation target_loc = _grid->relative_location(actor->location, static_cast<Orientation>(actor->orientation));

    // Check if target location is within grid bounds
    if (!is_valid_location(target_loc)) {
      return false;
    }

    // First try to find an object at the agent layer
    GridObject* obj = safe_object_at(target_loc);

    // If nothing found at default layer, try the object layer
    if (obj == nullptr) {
      target_loc.layer = GridLayer::Object_Layer;
      obj = safe_object_at(target_loc);
    }

    // If still nothing found, return false
    if (obj == nullptr) {
      return false;
    }

    // Use dynamic_cast for type safety
    MettaObject* target = dynamic_cast<MettaObject*>(obj);
    if (target == nullptr) {
      throw std::runtime_error("Object at target location is not a MettaObject");
    }

    // Check if the object is swappable
    if (!target->swappable()) {
      return false;  // Not swappable is a normal gameplay situation
    }

    // Validate type ID for stats tracking
    if (target->_type_id >= ObjectType::Count) {
      throw std::runtime_error("Invalid object type ID: " + std::to_string(target->_type_id));
    }

    // Track the swap in stats
    actor->stats.incr("swap", _stats.target[target->_type_id]);

    // Perform the swap
    _grid->swap_objects(actor->id, target->id);
    return true;
  }
};
}  // namespace Actions
#endif  // SWAP_HPP