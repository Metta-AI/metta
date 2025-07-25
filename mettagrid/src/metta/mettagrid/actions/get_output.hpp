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
      : ActionHandler(cfg, "get_output"), blue_battery_item_(blue_battery_item) {}

  unsigned char max_arg() const override {
    return 0;
  }

protected:
  bool _handle_action(Agent* actor, ActionArg /*arg*/) override {
    std::ofstream debug_log("/tmp/mettagrid_get_output_debug.log", std::ios_base::app);
    std::ostringstream oss;
    oss << "[GetOutput] Agent " << (actor ? std::to_string(actor->agent_id) : "nullptr") << " at ("
        << (actor ? std::to_string(actor->location.r) : "?") << ", "
        << (actor ? std::to_string(actor->location.c) : "?") << ")\n";
    debug_log << oss.str();

    GridLocation target_loc = _grid->relative_location(actor->location, static_cast<Orientation>(actor->orientation));
    target_loc.layer = GridLayer::ObjectLayer;
    debug_log << "  Target location: (" << target_loc.r << ", " << target_loc.c << ", layer=" << int(target_loc.layer) << ")\n";

    // Default is taking output from converter
    Converter* converter = dynamic_cast<Converter*>(_grid->object_at(target_loc));
    if (!converter) {
      debug_log << "  No converter at target. blue_battery_item_=" << blue_battery_item_ << "\n";
      if (blue_battery_item_ < 0){
        debug_log << "  blue_battery_item_ < 0, returning false.\n";
        debug_log.close();
        return false;
      }

      // If Box, handle special logic
      Box* box = dynamic_cast<Box*>(_grid->object_at(target_loc));
      if (box) {
        debug_log << "  Found box at target. box->creator_agent_id=" << int(box->creator_agent_id)
                  << ", actor->agent_id=" << int(actor->agent_id) << "\n";
        // if (actor->agent_id == box->creator_agent_id) {
        //   // Creator cannot open their own box
        //   return false;
        // }
        // Direct lookup using creator_agent_object_id
        Agent* creator = dynamic_cast<Agent*>(_grid->object(box->creator_agent_object_id));
        debug_log << "  Creator agent object id: " << int(box->creator_agent_object_id)
                  << ", creator ptr: " << (creator ? "valid" : "nullptr") << "\n";
        if (!creator) {
          debug_log << "  Creator agent not found. Returning false.\n";
          debug_log.close();
          return false;
        }
        debug_log << "  Returning blue battery to creator.\n";
        // Return blue battery to creator
        creator->update_inventory(blue_battery_item_, 1);
        // debug_log << "  Removing box from grid.\n";
        // // Remove box from grid
        _grid->remove_object(box);
        debug_log << "Calling remove_object...\n";
        // auto removed = _grid->remove_object(box);
        // debug_log << "remove_object returned. removed ptr: " << removed.get() << "\n";
        // debug_log << "objects.size()=" << _grid->objects.size() << "\n";
        // debug_log << "About to remove box: ptr=" << box
        //   << " id=" << int(box->id)
        //   << " grid->object(box->id)=" << _grid->object(box->id)
        //   << " (should match ptr)\n";
        // debug_log << "Box location: (" << int(box->location.r) << "," << int(box->location.c) << "," << int(box->location.layer) << ")\n";
        // debug_log << "Object at location: " << _grid->object_at(box->location) << "\n";
        // debug_log << "  Adding stats for box opened and battery returned.\n";
        actor->stats.add("box.opened", 1.0f);
        debug_log << "  Adding stats for box opened and battery returned.\n";
        debug_log.close();
        return true;
      }
      debug_log << "  No box at target. Returning false.\n";
      debug_log.close();
      return false;
    }

    debug_log << "  Found converter at target. Checking inventory_is_accessible...\n";
    if (!converter->inventory_is_accessible()) {
      debug_log << "  Converter inventory is not accessible. Returning false.\n";
      debug_log.close();
      return false;
    }

    // Actions is only successful if we take at least one item.
    bool resources_taken = false;
    debug_log << "  Attempting to take resources from converter.\n";
    for (const auto& [item, _] : converter->output_resources) {
      debug_log << "    Checking item " << int(item) << ": ";
      if (converter->inventory.count(item) == 0) {
        debug_log << "not present in inventory.\n";
        continue;
      }
      InventoryDelta resources_available = static_cast<InventoryDelta>(converter->inventory[item]);
      debug_log << resources_available << " available. ";
      InventoryDelta taken = actor->update_inventory(item, resources_available);
      debug_log << "Agent took " << taken << ". ";
      if (taken > 0) {
        debug_log << "Success. Updating converter inventory.\n";
        actor->stats.add(actor->stats.inventory_item_name(item) + ".get", static_cast<float>(taken));
        debug_log << "Success. Updating converter inventory.\n";
        converter->update_inventory(item, -taken);
        resources_taken = true;
      } else {
        debug_log << "Nothing taken.\n";
      }
    }
    debug_log << "  Done attempting to take resources. resources_taken=" << resources_taken << "\n";
    debug_log.close();
    return resources_taken;
  }
};

#endif  // ACTIONS_GET_OUTPUT_HPP_
