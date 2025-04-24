#ifndef PUT_RECIPE_ITEMS_HPP
#define PUT_RECIPE_ITEMS_HPP

#include <string>

#include "action_handler.hpp"
#include "grid.hpp"
#include "grid_object.hpp"
#include "objects/agent.hpp"
#include "objects/converter.hpp"

class PutRecipeItems : public ActionHandler {
public:
  PutRecipeItems(const ActionConfig& cfg) : ActionHandler(cfg, "put_recipe_items") {}

  unsigned char max_arg() const override {
    return 0;
  }

protected:
  bool _handle_action(unsigned int actor_id, Agent* actor, ActionArg arg) override {
    GridLocation target_loc = _grid->relative_location(actor->location, static_cast<Orientation>(actor->orientation));
    target_loc.layer = GridLayer::Object_Layer;
    MettaObject* target = static_cast<MettaObject*>(_grid->object_at(target_loc));
    if (target == nullptr || !target->has_inventory()) {
      return false;
    }

    // #Converter_and_HasInventory_are_the_same_thing
    Converter* converter = static_cast<Converter*>(target);

    for (size_t i = 0; i < converter->recipe_input.size(); i++) {
      if (converter->recipe_input[i] > actor->inventory[i]) {
        return false;
      }
    }

    for (size_t i = 0; i < converter->recipe_input.size(); i++) {
      actor->update_inventory(static_cast<InventoryItem>(i), -converter->recipe_input[i]);
      converter->update_inventory(static_cast<InventoryItem>(i), converter->recipe_input[i]);
      actor->stats.add(InventoryItemNames[i], "put", converter->recipe_input[i]);
    }

    return true;
  }
};

#endif  // PUT_RECIPE_ITEMS_HPP
