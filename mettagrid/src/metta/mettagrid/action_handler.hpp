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

struct ActionTrackingState {
  std::string last_action_name;
  unsigned int consecutive_count = 0;
  unsigned int total_count = 0;
  bool last_success = false;
  unsigned int current_step = 0;
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

    // Update tracking for this agent
    update_tracking(actor->agent_id, success);

    // Track success/failure
    if (success) {
      actor->stats.incr("action." + _action_name + ".success");
      for (const auto& [item, amount] : _consumed_resources) {
        InventoryDelta delta = actor->update_inventory(item, -static_cast<InventoryDelta>(amount));
        // We consume resources after the action succeeds, but in the future
        // we might have an action that uses the resource. This check will
        // catch that.
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

  // Get tracking state for a specific agent
  const ActionTrackingState* get_agent_tracking(size_t agent_id) const {
    auto it = _agent_tracking.find(agent_id);
    return it != _agent_tracking.end() ? &it->second : nullptr;
  }

  // Get the last action name for an agent across all handlers
  static std::string get_last_action_name(size_t agent_id) {
    auto& actions = get_global_last_actions();
    auto it = actions.find(agent_id);
    return it != actions.end() ? it->second : "";
  }

  // Clear all tracking data (useful for reset)
  static void clear_all_tracking() {
    get_global_last_actions().clear();
  }

  // Clear tracking for this handler
  void clear_tracking() {
    _agent_tracking.clear();
  }

protected:
  virtual bool _handle_action(Agent* actor, ActionArg arg) = 0;

  std::string _action_name;
  std::map<InventoryItem, InventoryQuantity> _required_resources;
  std::map<InventoryItem, InventoryQuantity> _consumed_resources;

  // Per-agent tracking state for this handler
  std::map<size_t, ActionTrackingState> _agent_tracking;

  // REMOVED: static std::map<size_t, std::string> _global_last_actions;

private:
  // Use function-local static to avoid global destructor
  static std::map<size_t, std::string>& get_global_last_actions() {
    static std::map<size_t, std::string> global_last_actions;
    return global_last_actions;
  }

  void update_tracking(size_t agent_id, bool success) {
    auto& state = _agent_tracking[agent_id];
    auto& global_actions = get_global_last_actions();

    // Check if this is consecutive
    if (global_actions[agent_id] == _action_name && success) {
      state.consecutive_count++;
    } else {
      state.consecutive_count = success ? 1 : 0;
    }

    state.last_action_name = _action_name;
    state.last_success = success;
    if (success) {
      state.total_count++;
    }

    // Update global tracking
    if (success) {
      global_actions[agent_id] = _action_name;
    }
  }
};

#endif  // ACTION_HANDLER_HPP_
