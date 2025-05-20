#ifndef PUT_RECIPE_ITEMS_HPP
#define PUT_RECIPE_ITEMS_HPP

#include <cstdint>
#include <string>

#include "actions/action_handler.hpp"
#include "grid.hpp"
#include "grid_object.hpp"
#include "objects/agent.hpp"
#include "objects/converter.hpp"

namespace Actions {

class PutRecipeItems : public ActionHandler {
public:
  PutRecipeItems(const ActionConfig& cfg) : ActionHandler(cfg, "put_recipe_items") {}

  uint8_t max_arg() const override {
    return 0;
  }

  ActionHandler* clone() const override {
    return new PutRecipeItems(*this);
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

    target_loc.layer = GridLayer::Object_Layer;

    // Use safe_object_at instead of direct access
    GridObject* obj = safe_object_at(target_loc);
    if (obj == nullptr) {
      return false;
    }

    // Use dynamic_cast for type safety
    MettaObject* target = dynamic_cast<MettaObject*>(obj);
    if (target == nullptr) {
      throw std::runtime_error("Object at target location is not a MettaObject");
    }

    if (!target->is_converter()) {
      return false;
    }

    Converter* converter = dynamic_cast<Converter*>(target);
    if (converter == nullptr) {
      throw std::runtime_error("Object with is_converter() is not a Converter");
    }

    // Check if we have enough items for the recipe
    for (uint32_t i = 0; i < converter->recipe_input.size(); i++) {
      // Validate inventory item index
      if (i >= InventoryItem::InventoryCount) {
        throw std::runtime_error("Recipe input index out of range: " + std::to_string(i));
      }

      // for (size_t i = 0; i < converter->recipe_input.size(); i++) {
      //   if (converter->recipe_input[i] > actor->inventory[i]) {
      //     return false;
      //   }
      // }

      bool success = false;
      for (size_t i = 0; i < converter->recipe_input.size(); i++) {
        unsigned int inv = std::min(converter->recipe_input[i], actor->inventory[i]);
        if (inv == 0) {
          continue;
        }
        actor->update_inventory(static_cast<InventoryItem>(i), -inv);
        converter->update_inventory(static_cast<InventoryItem>(i), inv);
        actor->stats.add(InventoryItemNames[i], "put", inv);
        success = true;
      }

      return success;
    }
  };
}  // namespace Actions
#endif  // PUT_RECIPE_ITEMS_HPP