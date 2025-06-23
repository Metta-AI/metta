#ifndef METTAGRID_METTAGRID_ACTIONS_GET_OUTPUT_HPP_
#define METTAGRID_METTAGRID_ACTIONS_GET_OUTPUT_HPP_

#include <string>

#include "action_handler.hpp"
#include "grid.hpp"
#include "grid_object.hpp"
#include "objects/agent.hpp"
#include "objects/converter.hpp"
#include "types.hpp"

class GetOutput : public ActionHandler {
public:
  explicit GetOutput(const ActionConfig& cfg) : ActionHandler(cfg, "get_output") {}

  unsigned char max_arg() const override {
    return 0;
  }

protected:
  bool _handle_action(Agent* actor, ActionArg arg) override {
    GridLocation target_loc = _grid->relative_location(actor->location, static_cast<Orientation>(actor->orientation));
    target_loc.layer = GridLayer::Object_Layer;
    // get_output only works on Converters, since only Converters have an output.
    // Once we generalize this to `get`, we should be able to get from any HasInventory object, which
    // should include agents. That's (e.g.) why we're checking inventory_is_accessible.
    Converter* converter = dynamic_cast<Converter*>(_grid->object_at(target_loc));
    if (converter == nullptr) {
      return false;
    }

    if (!converter->inventory_is_accessible()) {
      return false;
    }

    // Actions is only successful if we take at least one item.
    bool items_taken = false;

    for (size_t i = 0; i < InventoryItem::InventoryItemCount; i++) {
      if (converter->recipe_output[i] == 0) {
        // We only want to take things the converter can produce. Otherwise it's a pain to
        // collect resources from a converter that's in the middle of processing a queue.
        continue;
      }

      int taken = actor->update_inventory(static_cast<InventoryItem>(i), converter->inventory[i]);

      if (taken > 0) {
        actor->stats.add(InventoryItemNames[i] + ".get", taken);
        converter->update_inventory(static_cast<InventoryItem>(i), -taken);
        items_taken = true;
      }
    }

    return items_taken;
  }
};

#endif  // METTAGRID_METTAGRID_ACTIONS_GET_OUTPUT_HPP_
