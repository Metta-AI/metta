#ifndef ACTIONS_PUT_RECIPE_ITEMS_HPP_
#define ACTIONS_PUT_RECIPE_ITEMS_HPP_

#include <string>

#include "action_handler.hpp"
#include "grid.hpp"
#include "grid_object.hpp"
#include "objects/agent.hpp"
#include "objects/converter.hpp"
#include "types.hpp"

class PutRecipeItems : public ActionHandler {
public:
  explicit PutRecipeItems(const ActionConfig& cfg) : ActionHandler(cfg, "put_recipe_items") {}

  unsigned char max_arg() const override {
    return 0;
  }

protected:
  bool _handle_action(Agent* actor, ActionArg arg) override {
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
    for (const auto& [item, amount] : converter->recipe_input) {
      if (actor->inventory.count(item) == 0) {
        continue;
      }
      int max_to_put = std::min(amount, actor->inventory.at(item));
      if (max_to_put > 0) {
        int put = converter->update_inventory(item, max_to_put);
        if (put > 0) {
          // We should be able to put this many items into the converter. If not, something is wrong.
          int delta = actor->update_inventory(item, -put);
          assert(delta == -put);
          actor->stats.add(actor->stats.inventory_item_name(item) + ".put", put);
          success = true;
        }
      }
    }

    return success;
  }
};

#endif  // ACTIONS_PUT_RECIPE_ITEMS_HPP_
