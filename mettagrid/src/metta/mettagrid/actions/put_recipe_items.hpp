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
  bool _handle_action(Agent* actor, [[maybe_unused]] ActionArg arg) override {
    GridLocation target_loc = _grid->relative_location(actor->location, static_cast<Orientation>(actor->orientation));
    target_loc.layer = GridLayer::ObjectLayer;
    // put_recipe_items only works on Converters, since only Converters have a recipe.
    // Once we generalize this to `put`, we should be able to put to any HasInventory object, which
    // should include agents.
    Converter* converter = dynamic_cast<Converter*>(_grid->object_at(target_loc));
    if (converter == nullptr) {
      return false;
    }

    bool success = false;
    for (const auto& [item, items_required] : converter->input_resources) {
      if (actor->inventory.count(item) == 0) {
        continue;
      }
      InventoryQuantity items_available = actor->inventory.at(item);
      InventoryQuantity items_to_put = std::min(items_required, items_available);

      if (items_to_put > 0) {
        InventoryDelta items_put = converter->update_inventory(item, static_cast<InventoryDelta>(items_to_put));
        if (items_put > 0) {
          InventoryDelta delta = actor->update_inventory(item, -items_put);
          assert(delta == -items_put);
          actor->stats.add(actor->stats.inventory_item_name(item) + ".put", items_put);
          success = true;
        }
      }
    }

    return success;
  }
};

#endif  // ACTIONS_PUT_RECIPE_ITEMS_HPP_
