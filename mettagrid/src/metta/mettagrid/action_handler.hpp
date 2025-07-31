#ifndef ACTION_HANDLER_HPP_
#define ACTION_HANDLER_HPP_

#include <map>
#include <string>
#include <vector>

#include "grid.hpp"
#include "grid_object.hpp"
#include "objects/agent.hpp"
#include "objects/constants.hpp"
#include "types.hpp"

struct ActionConfig {
  std::map<InventoryItem, InventoryQuantity> required_resources;
  std::map<InventoryItem, InventoryQuantity> consumed_resources;

  ActionConfig(const std::map<InventoryItem, InventoryQuantity>& required_resources,
               const std::map<InventoryItem, InventoryQuantity>& consumed_resources)
      : required_resources(required_resources), consumed_resources(consumed_resources) {}

  virtual ~ActionConfig() {}
};

class ActionHandler {
public:
  unsigned char priority;
  Grid* _grid{};

  ActionHandler(const ActionConfig& cfg, const std::string& action_name)
      : priority(0),
        _action_name(action_name),
        _required_resources(cfg.required_resources),
        _consumed_resources(cfg.consumed_resources) {
    for (const auto& [item, amount] : _required_resources) {
      if (amount < _consumed_resources[item]) {
        throw std::runtime_error("Required resources must be greater than or equal to consumed resources");
      }
    }
  }

  virtual ~ActionHandler() {}

  void init(Grid* grid) {
    this->_grid = grid;
  }

  bool handle_action(GridObjectId actor_object_id, ActionArg arg) {
    Agent* actor = static_cast<Agent*>(_grid->object(actor_object_id));

    // Handle frozen status
    if (actor->frozen != 0) {
      actor->stats.incr("status.frozen.ticks");
      actor->stats.incr("status.frozen.ticks." + actor->group_name);
      if (actor->frozen > 0) {
        actor->frozen -= 1;
      }
      return false;
    }

    bool has_needed_resources = true;
    for (const auto& [item, amount] : _required_resources) {
      if (actor->inventory[item] < amount) {
        has_needed_resources = false;
        break;
      }
    }

    // Execute the action
    bool success = has_needed_resources && _handle_action(actor, arg);

    // The intention here is to provide a metric that reports when an agent has stayed in one location for a long
    // period, perhaps spinning in circles. We think this could be a good indicator that a policy has collapsed.
    if (actor->location == actor->prev_location) {
      actor->steps_without_motion += 1;
      if (actor->steps_without_motion > actor->stats.get("status.max_steps_without_motion")) {
        actor->stats.set("status.max_steps_without_motion", actor->steps_without_motion);
      }
    } else {
      actor->steps_without_motion = 0;
    }

    // Update tracking for this agent
    actor->prev_action_name = _action_name;
    actor->prev_location = actor->location;

    // Track success/failure
    if (success) {
      actor->stats.incr("action." + _action_name + ".success");
      for (const auto& [item, amount] : _consumed_resources) {
        InventoryDelta delta = actor->update_inventory(item, -static_cast<InventoryDelta>(amount));
        // We consume resources after the action succeeds, but in the future we might have an action that uses the
        // resource. This check will catch that.
        assert(delta == -amount);
      }
    } else {
      actor->stats.incr("action." + _action_name + ".failed");
      actor->stats.incr("action.failure_penalty");
      *actor->reward -= actor->action_failure_penalty;
    }

    return success;
  }

  virtual unsigned char max_arg() const {
    return 0;
  }

  std::string action_name() const {
    return _action_name;
  }

protected:
  virtual bool _handle_action(Agent* actor, ActionArg arg) = 0;

  std::string _action_name;
  std::map<InventoryItem, InventoryQuantity> _required_resources;
  std::map<InventoryItem, InventoryQuantity> _consumed_resources;
};

#endif  // ACTION_HANDLER_HPP_
