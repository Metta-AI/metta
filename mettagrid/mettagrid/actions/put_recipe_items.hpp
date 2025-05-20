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
    // put_recipe_items only works on Converters, since only Converters have a recipe.
    // Once we generalize this to `put`, we should be able to put to any HasInventory object, which
    // should include agents.
    Converter* converter = dynamic_cast<Converter*>(_grid->object_at(target_loc));
    if (converter == nullptr) {
      return false;
    }

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

#endif  // PUT_RECIPE_ITEMS_HPP
