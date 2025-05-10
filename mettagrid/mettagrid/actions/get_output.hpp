#ifndef GET_OUTPUT_HPP
#define GET_OUTPUT_HPP

#include <cstdint>
#include <string>

#include "actions/action_handler.hpp"
#include "grid.hpp"
#include "grid_object.hpp"
#include "objects/agent.hpp"
#include "objects/converter.hpp"
namespace Actions {
class GetOutput : public ActionHandler {
public:
  GetOutput(const ActionConfig& cfg) : ActionHandler(cfg, "get_output") {}

  uint8_t max_arg() const override {
    return 0;
  }

  ActionHandler* clone() const override {
    return new GetOutput(*this);
  }

protected:
  bool _handle_action(uint32_t actor_id, Agent* actor, c_actions_type arg) override {
    // Validate orientation
    validate_orientation(actor);

    // Get target location
    GridLocation target_loc = _grid->relative_location(actor->location, static_cast<Orientation>(actor->orientation));

    // Check if target location is within grid bounds
    if (!is_valid_location(target_loc)) {
      return false;
    }

    target_loc.layer = GridLayer::Object_Layer;

    GridObject* obj = safe_object_at(target_loc);
    if (obj == nullptr) {
      return false;
    }

    // Use dynamic_cast for type safety
    MettaObject* target = dynamic_cast<MettaObject*>(obj);
    if (target == nullptr) {
      throw std::runtime_error("Object at target location is not a MettaObject");
    }

    if (!target->has_inventory()) {
      return false;
    }

    // TODO: figure this out -- agents have inventory?!
    //
    // ### Converter and HasInventory are the same thing ###
    // It's more correct to cast this as a HasInventory, but right now Converters are
    // the only implementors of HasInventory, and we also need to call maybe_start_converting
    // on them. We should later refactor this to we call .update_inventory on the target, and
    // have this automatically call maybe_start_converting. That's hard because we need to
    // let it maybe schedule events.
    Converter* converter = dynamic_cast<Converter*>(target);
    if (converter == nullptr) {
      throw std::runtime_error("Object with has_inventory() is not a Converter");
    }

    if (!converter->inventory_is_accessible()) {
      return false;
    }

    // Actions is only successful if we take at least one item.
    bool items_taken = false;

    for (uint32_t i = 0; i < InventoryItem::InventoryCount; i++) {
      if (converter->recipe_output[i] == 0) {
        // We only want to take things the converter can produce. Otherwise it's a pain to
        // collect resources from a converter that's in the middle of processing a queue.
        continue;
      }

      // Only take resources if the converter has some.
      if (converter->inventory[i] > 0) {
        // Validate the inventory item index
        if (i >= InventoryItemNames.size()) {
          throw std::runtime_error("Invalid inventory item index: " + std::to_string(i));
        }

        // The actor will destroy anything it can't hold. That's not intentional, so feel free
        // to fix it.
        actor->stats.add(InventoryItemNames[i], "get", converter->inventory[i]);
        actor->update_inventory(static_cast<InventoryItem>(i), converter->inventory[i]);
        converter->update_inventory(static_cast<InventoryItem>(i), -converter->inventory[i]);
        items_taken = true;
      }
    }

    return items_taken;
  }
};
}  // namespace Actions

#endif  // GET_OUTPUT_HPP