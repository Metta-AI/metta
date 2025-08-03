#ifndef ACTIONS_GET_OUTPUT_HPP_
#define ACTIONS_GET_OUTPUT_HPP_

#include <string>
#include <fstream>
#include <sstream>
#include <iomanip>

#include "action_handler.hpp"
#include "grid.hpp"
#include "grid_object.hpp"
#include "objects/agent.hpp"
#include "objects/converter.hpp"
#include "objects/box.hpp"
#include "types.hpp"

class GetOutput : public ActionHandler {
public:
  int blue_battery_item_;
  explicit GetOutput(const ActionConfig& cfg, int blue_battery_item)
      : ActionHandler(cfg, "get_items"), blue_battery_item_(blue_battery_item) {}

  unsigned char max_arg() const override {
    return 0;
  }

protected:
  bool _handle_action(Agent* actor, ActionArg /*arg*/) override {

    GridLocation target_loc = _grid->relative_location(actor->location, static_cast<Orientation>(actor->orientation));
    target_loc.layer = GridLayer::ObjectLayer;

    // Default is taking output from converter
    Converter* converter = dynamic_cast<Converter*>(_grid->object_at(target_loc));
    if (!converter) {
      // no blue batteries in the game, don't process potential for picking up box
      if (blue_battery_item_ < 0){
        return false;
      }

      // If Box, handle special logic
      Box* box = dynamic_cast<Box*>(_grid->object_at(target_loc));
      if (box) {
        if (actor->agent_id == box->creator_agent_id) {
          // Creator cannot open their own box
          return false;
        }
        // If the creator of the box is an agent, return blue battery to creator and penalize creator
        if (box->creator_agent_id != 255) {
          Agent* creator = dynamic_cast<Agent*>(_grid->object(box->creator_agent_object_id));
          if (!creator) {
            return false;
          }
          // Return blue battery to creator
          creator->update_inventory(blue_battery_item_, 1);
          if (creator->reward) *creator->reward -= 1.0f;
        }

        // Reward the agent for opening the box and teleport back to top-left corner
        if (actor->reward) *actor->reward += 1.0f;
        _grid->move_object(box->id, GridLocation(0, 0, GridLayer::ObjectLayer));
        actor->box->inventory[blue_battery_item_] = 0;
        actor->stats.add("box.opened", 1.0f);
        return true;
      }
      return false;
    }

    if (!converter->inventory_is_accessible()) {
      return false;
    }

    // Actions is only successful if we take at least one item.
    bool resources_taken = false;
    for (const auto& [item, _] : converter->output_resources) {
      if (converter->inventory.count(item) == 0) {
        continue;
      }
      InventoryDelta resources_available = static_cast<InventoryDelta>(converter->inventory[item]);
      InventoryDelta taken = actor->update_inventory(item, resources_available);
      if (taken > 0) {
        actor->stats.add(actor->stats.inventory_item_name(item) + ".get", static_cast<float>(taken));
        converter->update_inventory(item, -taken);
        resources_taken = true;
      }
    }
    return resources_taken;
  }
};

#endif  // ACTIONS_GET_OUTPUT_HPP_
