#ifndef PACKAGES_METTAGRID_CPP_INCLUDE_METTAGRID_ACTIONS_GET_OUTPUT_HPP_
#define PACKAGES_METTAGRID_CPP_INCLUDE_METTAGRID_ACTIONS_GET_OUTPUT_HPP_

#include <string>

#include "actions/action_handler.hpp"
#include "core/grid.hpp"
#include "core/grid_object.hpp"
#include "core/types.hpp"
#include "objects/agent.hpp"
#include "objects/converter.hpp"

class GetOutput : public ActionHandler {
public:
  explicit GetOutput(const ActionConfig& cfg) : ActionHandler(cfg, "get_items") {}

  unsigned char max_arg() const override {
    return 0;
  }

protected:
  bool _handle_action(Agent& actor, ActionArg /*arg*/) override {
    GridLocation target_loc = _grid->relative_location(actor.location, static_cast<Orientation>(actor.orientation));
    target_loc.layer = GridLayer::ObjectLayer;
    // get_output only works on Converters, since only Converters have an output.
    // Once we generalize this to `get`, we should be able to get from any HasInventory object, which
    // should include agents. That's (e.g.) why we're checking inventory_is_accessible.
    Converter* converter = dynamic_cast<Converter*>(_grid->object_at(target_loc));

    if (converter) {
      if (!converter->inventory_is_accessible()) {
        return false;
      }

      // Actions is only successful if we take at least one item.
      bool resources_taken = false;

      for (const auto& [item, _] : converter->output_resources) {
        InventoryDelta resources_available = converter->inventory.amount(item);
        InventoryDelta taken = actor.update_inventory(item, resources_available);

        if (taken > 0) {
          actor.stats.add(actor.stats.resource_name(item) + ".get", taken);
          converter->update_inventory(item, -taken);
          resources_taken = true;
        }
      }
      return resources_taken;
    }
    return false;
  }
};

#endif  // PACKAGES_METTAGRID_CPP_INCLUDE_METTAGRID_ACTIONS_GET_OUTPUT_HPP_
