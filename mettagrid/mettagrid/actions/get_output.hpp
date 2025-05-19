#ifndef GET_OUTPUT_HPP
#define GET_OUTPUT_HPP

#include <string>

#include "action_handler.hpp"
#include "grid.hpp"
#include "grid_object.hpp"
#include "objects/agent.hpp"
#include "objects/converter.hpp"

class GetOutput : public ActionHandler {
public:
  GetOutput(const ActionConfig& cfg) : ActionHandler(cfg, "get_output") {}

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

    // ##Converter_and_HasInventory_are_the_same_thing
    // It's more correct to cast this as a HasInventory, but right now Converters are
    // the only implementors of HasInventory, and we also need to call maybe_start_converting
    // on them. We should later refactor this to we call .update_inventory on the target, and
    // have this automatically call maybe_start_converting. That's hard because we need to
    // let it maybe schedule events.
    Converter* converter = static_cast<Converter*>(target);
    if (!converter->inventory_is_accessible()) {
      return false;
    }

    // Actions is only successful if we take at least one item.
    bool items_taken = false;

    for (size_t i = 0; i < InventoryItem::InventoryCount; i++) {
      if (converter->recipe_output[i] == 0) {
        // We only want to take things the converter can produce. Otherwise it's a pain to
        // collect resources from a converter that's in the middle of processing a queue.
        continue;
      }
      unsigned char can_take = std::min<unsigned char>(actor->max_items - actor->inventory[i], converter->inventory[i]);

      if (can_take > 0) {
        actor->stats.add(InventoryItemNames[i], "get", can_take);
        actor->update_inventory(static_cast<InventoryItem>(i), can_take);
        converter->update_inventory(static_cast<InventoryItem>(i), -can_take);
        items_taken = true;
      }
    }

    return items_taken;
  }
};

#endif  // GET_OUTPUT_HPP
