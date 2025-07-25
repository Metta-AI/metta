#ifndef ACTIONS_PUT_RECIPE_ITEMS_HPP_
#define ACTIONS_PUT_RECIPE_ITEMS_HPP_

#include <string>

#include "action_handler.hpp"
#include "grid.hpp"
#include "grid_object.hpp"
#include "objects/agent.hpp"
#include "objects/converter.hpp"
#include "objects/box.hpp"
#include "types.hpp"

class PutRecipeItems : public ActionHandler {
public:
  int blue_battery_item_;
  TypeId box_type_id_;
  const std::string& box_type_name_;
  explicit PutRecipeItems(const ActionConfig& cfg, TypeId box_type_id, const std::string& box_type_name, int blue_battery_item)
      : ActionHandler(cfg, "put_recipe_items"), blue_battery_item_(blue_battery_item),
      box_type_id_(box_type_id), box_type_name_(box_type_name) {}

  unsigned char max_arg() const override {
    return 0;
  }

protected:
  bool _handle_action(Agent* actor, ActionArg /*arg*/) override {
    GridLocation target_loc = _grid->relative_location(actor->location, static_cast<Orientation>(actor->orientation));
    target_loc.layer = GridLayer::ObjectLayer;

    // Default is putting to converter
    Converter* converter = dynamic_cast<Converter*>(_grid->object_at(target_loc));
    if (!converter) {
      // If we are putting to a non-converter, the only option is we are trying to create a box
      // Box creation: use 1 blue battery to create a box
      // Only allow box creation if target location is empty
      // if (_grid->is_empty(target_loc.r, target_loc.c) && blue_battery_item_ >= 0) {
      //   if (actor->inventory.count(blue_battery_item_) && actor->inventory[blue_battery_item_] > 0) {
      //     actor->update_inventory(blue_battery_item_, -1);
      //     Box* box = new Box(target_loc.r, target_loc.c, /*type_id=*/100, "box", actor->id, actor->agent_id, blue_battery_item_);
      //     _grid->add_object(box);
      //     actor->stats.add("box.created", 1.0f);
      //     return true;
      //   }
      // }
      if (_grid->is_empty(target_loc.r, target_loc.c) && blue_battery_item_ >= 0) {
        if (actor->inventory.count(blue_battery_item_) && actor->inventory[blue_battery_item_] > 0) {
          actor->update_inventory(blue_battery_item_, -1);
          actor->how_long_blue_battery_held = 0;
          Box* box = new Box(target_loc.r, target_loc.c, box_type_id_, box_type_name_, actor->id, actor->agent_id, blue_battery_item_);
          _grid->add_object(box);
          actor->stats.add("box.created", 1.0f);
          return true;
        }
      }

      return false;
    }

    bool success = false;
    for (const auto& [item, resources_required] : converter->input_resources) {
      if (actor->inventory.count(item) == 0) {
        continue;
      }
      InventoryQuantity resources_available = actor->inventory.at(item);
      InventoryQuantity resources_to_put = std::min(resources_required, resources_available);

      if (resources_to_put > 0) {
        InventoryDelta resources_put = converter->update_inventory(item, static_cast<InventoryDelta>(resources_to_put));
        if (resources_put > 0) {
          InventoryDelta delta = actor->update_inventory(item, -resources_put);
          assert(delta == -resources_put);
          actor->stats.add(actor->stats.inventory_item_name(item) + ".put", static_cast<float>(resources_put));
          success = true;
        }
      }
    }

    return success;
  }
};

#endif  // ACTIONS_PUT_RECIPE_ITEMS_HPP_
