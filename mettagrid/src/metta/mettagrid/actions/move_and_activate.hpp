#ifndef ACTIONS_MOVE_AND_ACTIVATE_HPP_
#define ACTIONS_MOVE_AND_ACTIVATE_HPP_

#include <string>

#include "action_handler.hpp"
#include "grid_object.hpp"
#include "objects/agent.hpp"
#include "objects/converter.hpp"
#include "objects/has_inventory.hpp"
#include "orientation.hpp"
#include "types.hpp"

// Forward declaration
struct GameConfig;

/**
 * MoveAndActivate action handler that combines movement with object activation.
 *
 * This action attempts to move the agent in a specified direction. If the movement
 * is blocked by an object (like a converter/building), it will attempt to activate
 * or interact with that object instead. This implements the "move-to-use" mechanic
 * commonly found in games where walking into an object activates it.
 *
 * Behavior:
 * 1. If target location is empty: Move the agent there
 * 2. If target location has a converter: Transfer resources to it
 * 3. If target location has another object: Action fails
 *
 * The agent's orientation is always updated to face the target direction,
 * regardless of whether the action succeeds.
 */
class MoveAndActivate : public ActionHandler {
public:
  explicit MoveAndActivate(const ActionConfig& cfg, const GameConfig* game_config)
      : ActionHandler(cfg, "move_and_activate"), _game_config(game_config) {}

  unsigned char max_arg() const override {
    return _game_config->allow_diagonals ? 7 : 3;  // 8 directions if diagonals, 4 otherwise
  }

protected:
  bool _handle_action(Agent* actor, ActionArg arg) override {
    // Get the orientation from the action argument
    Orientation move_direction = static_cast<Orientation>(arg);

    // Validate the direction based on diagonal support
    if (!isValidOrientation(move_direction, _game_config->allow_diagonals)) {
      return false;
    }

    GridLocation current_location = actor->location;
    GridLocation target_location = current_location;

    // Get movement deltas for the direction
    int dc, dr;
    getOrientationDelta(move_direction, dc, dr);

    // Calculate target location
    target_location.r = static_cast<GridCoord>(static_cast<int>(target_location.r) + dr);
    target_location.c = static_cast<GridCoord>(static_cast<int>(target_location.c) + dc);

    // Always update orientation to face the movement direction
    actor->orientation = move_direction;

    // Check if target location is valid
    if (!_grid->is_valid_location(target_location)) {
      return false;
    }

    // Check if we can move to the target location
    if (_can_move_to(target_location)) {
      // Location is empty, perform the move
      bool move_success = _grid->move_object(actor->id, target_location);
      if (move_success) {
        _log_action_stats(actor, "move", "success");
      }
      return move_success;
    }

    // Location is blocked, try to activate/interact with the object there
    return _try_activate_object(actor, target_location);
  }

private:
  const GameConfig* _game_config;

  /**
   * Check if the target location is valid and empty for movement
   */
  bool _can_move_to(const GridLocation& target_location) {
    // Check both object layer and agent layer for obstacles
    if (!_grid->is_empty_at_layer(target_location.r, target_location.c, GridLayer::ObjectLayer)) {
      return false;
    }
    if (!_grid->is_empty(target_location.r, target_location.c)) {
      return false;
    }
    return true;
  }

  /**
   * Try to activate or interact with an object at the target location
   */
  bool _try_activate_object(Agent* actor, const GridLocation& target_location) {
    // Check for object at the object layer
    GridLocation obj_location = target_location;
    obj_location.layer = GridLayer::ObjectLayer;

    GridObject* target_object = _grid->object_at(obj_location);
    if (!target_object) {
      // No object to activate
      return false;
    }

    // Try to interact with a converter (building)
    Converter* converter = dynamic_cast<Converter*>(target_object);
    if (converter) {
      return _interact_with_converter(actor, converter);
    }

    // Try to interact with any object that has inventory
    HasInventory* inventory_object = dynamic_cast<HasInventory*>(target_object);
    if (inventory_object && inventory_object->inventory_is_accessible()) {
      return _interact_with_inventory(actor, inventory_object);
    }

    // Object exists but we can't interact with it
    _log_action_stats(actor, "activate", "blocked_by_" + target_object->type_name);
    return false;
  }

  /**
   * Interact with a converter by transferring input resources to it
   */
  bool _interact_with_converter(Agent* actor, Converter* converter) {
    bool any_transferred = false;

    // Try to transfer recipe input items to the converter
    for (const auto& [item, resources_required] : converter->input_resources) {
      if (actor->inventory.count(item) == 0) {
        continue;
      }

      InventoryQuantity resources_available = actor->inventory.at(item);
      InventoryQuantity resources_to_transfer = std::min(resources_required, resources_available);

      if (resources_to_transfer > 0) {
        // Try to add resources to the converter
        InventoryDelta resources_added = converter->update_inventory(item, resources_to_transfer);

        if (resources_added > 0) {
          // Remove the transferred resources from the agent
          [[maybe_unused]] InventoryDelta delta = actor->update_inventory(item, -resources_added);
          assert(delta == -resources_added);

          // Log the transfer
          const std::string item_name = actor->stats.resource_name(item);
          actor->stats.add("action.move_and_activate.transfer." + item_name, resources_added);
          actor->stats.add("action.move_and_activate.transfer.to." + converter->type_name, resources_added);

          any_transferred = true;
        }
      }
    }

    // Also try to collect output resources from the converter
    bool any_collected = false;
    for (const auto& [item, _] : converter->output_resources) {
      if (converter->inventory.count(item) == 0) {
        continue;
      }

      InventoryDelta resources_available = converter->inventory[item];
      InventoryDelta taken = actor->update_inventory(item, resources_available);

      if (taken > 0) {
        converter->update_inventory(item, -taken);

        const std::string item_name = actor->stats.resource_name(item);
        actor->stats.add("action.move_and_activate.collect." + item_name, taken);
        actor->stats.add("action.move_and_activate.collect.from." + converter->type_name, taken);

        any_collected = true;
      }
    }

    if (any_transferred || any_collected) {
      _log_action_stats(actor, "activate", converter->type_name);
      return true;
    }

    // Couldn't transfer or collect anything
    _log_action_stats(actor, "activate", "no_valid_transfer");
    return false;
  }

  /**
   * Generic interaction with any object that has inventory
   */
  bool _interact_with_inventory(Agent* actor, HasInventory* inventory_object) {
    GridObject* obj = static_cast<GridObject*>(inventory_object);
    bool any_transferred = false;

    // Try to take items from the object's inventory
    for (const auto& [item, amount] : inventory_object->inventory) {
      if (amount > 0) {
        InventoryDelta taken = actor->update_inventory(item, amount);

        if (taken > 0) {
          inventory_object->update_inventory(item, -taken);

          const std::string item_name = actor->stats.resource_name(item);
          actor->stats.add("action.move_and_activate.take." + item_name, taken);
          actor->stats.add("action.move_and_activate.take.from." + obj->type_name, taken);

          any_transferred = true;
        }
      }
    }

    if (any_transferred) {
      _log_action_stats(actor, "activate", obj->type_name);
      return true;
    }

    return false;
  }

  /**
   * Log action statistics
   */
  void _log_action_stats(Agent* actor, const std::string& action_type, const std::string& result) {
    const std::string& group = actor->group_name;
    actor->stats.incr("action.move_and_activate." + action_type + "." + group + "." + result);
  }
};

#endif  // ACTIONS_MOVE_AND_ACTIVATE_HPP_