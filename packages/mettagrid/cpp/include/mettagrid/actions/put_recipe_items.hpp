#ifndef PACKAGES_METTAGRID_CPP_INCLUDE_METTAGRID_ACTIONS_PUT_RECIPE_ITEMS_HPP_
#define PACKAGES_METTAGRID_CPP_INCLUDE_METTAGRID_ACTIONS_PUT_RECIPE_ITEMS_HPP_

#include <string>

#include "actions/action_handler.hpp"
#include "core/grid.hpp"
#include "core/grid_object.hpp"
#include "core/types.hpp"
#include "objects/agent.hpp"
#include "objects/converter.hpp"

class PutRecipeItems : public ActionHandler {
public:
  explicit PutRecipeItems(const ActionConfig& cfg) : ActionHandler(cfg, "put_items") {}

  unsigned char max_arg() const override {
    return 0;
  }

protected:
  bool _handle_action(Agent& actor, ActionArg /*arg*/) override {
    GridLocation target_loc = _grid->relative_location(actor.location, static_cast<Orientation>(actor.orientation));
    target_loc.layer = GridLayer::ObjectLayer;
    // put_recipe_items only works on Converters, since only Converters have a recipe.
    // Once we generalize this to `put`, we should be able to put to any HasInventory object, which
    // should include agents.
    Converter* converter = dynamic_cast<Converter*>(_grid->object_at(target_loc));
    if (!converter) {
      return false;
    }

    bool success = false;
    for (const auto& [item, resources_required] : converter->input_resources) {
      InventoryQuantity resources_available = actor.inventory.amount(item);
      InventoryQuantity resources_to_put = std::min(resources_required, resources_available);
      InventoryDelta resources_put = converter->update_inventory(item, resources_to_put);
      if (resources_put > 0) {
        [[maybe_unused]] InventoryDelta delta = actor.update_inventory(item, -resources_put);
        assert(delta == -resources_put);
        actor.stats.add(actor.stats.resource_name(item) + ".put", resources_put);
        success = true;
      }
    }

    return success;
  }
};

#endif  // PACKAGES_METTAGRID_CPP_INCLUDE_METTAGRID_ACTIONS_PUT_RECIPE_ITEMS_HPP_
