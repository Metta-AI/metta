#ifndef ACTION_HANDLER_HPP
#define ACTION_HANDLER_HPP

#include <cstdint>
#include <fstream>
#include <map>
#include <stdexcept>
#include <string>
#include <vector>

#include "constants.hpp"
#include "grid.hpp"
#include "grid_object.hpp"
#include "objects/agent.hpp"
#include "types.hpp"

struct StatNames {
  std::string success;
  std::string first_use;
  std::string failure;
  std::map<TypeId, std::string> target;
  std::map<TypeId, std::string> target_first_use;
  std::vector<std::string> group;
};

class ActionHandler {
public:
  uint8_t priority;
  Grid* _grid;

  ActionHandler(const ActionConfig& cfg, const std::string& action_name)
      : priority(0), _grid(nullptr), _action_name(action_name) {
    _stats.success = "action." + action_name;
    _stats.failure = "action." + action_name + ".failed";
    _stats.first_use = "action." + action_name + ".first_use";

    for (TypeId t = 0; t < ObjectType::Count; t++) {
      _stats.target[t] = _stats.success + "." + ObjectTypeNames[t];
      _stats.target_first_use[t] = _stats.first_use + "." + ObjectTypeNames[t];
    }
  }

  virtual ~ActionHandler() {}  // Virtual destructor

  void init(Grid* grid) {
    if (grid == nullptr) {
      throw std::runtime_error("Null grid passed to " + _action_name + "::init");
    }
    this->_grid = grid;
  }

  bool handle_action(uint32_t actor_id, GridObjectId actor_object_id, c_actions_type arg, uint32_t current_timestep) {
    std::ofstream debug_log("mettagrid_debug.log", std::ios::app);
    debug_log << "=== in handle action ===" << std::endl;

    // Validate grid initialization
    if (_grid == nullptr) {
      throw std::runtime_error("Grid not initialized in " + _action_name + " handler");
    }

    // Get the agent object
    GridObject* obj = _grid->object(actor_object_id);
    if (obj == nullptr) {
      throw std::runtime_error("Agent with ID " + std::to_string(actor_object_id) + " not found");
    }

    // Make sure it's an agent
    Agent* actor = dynamic_cast<Agent*>(obj);
    if (actor == nullptr) {
      throw std::runtime_error("Object with ID " + std::to_string(actor_object_id) + " is not an Agent in " +
                               _action_name + " handler");
    }
    if (&(actor->stats) == nullptr) {  // Check if stats object is accessible
      throw std::runtime_error("Agent with ID " + std::to_string(actor_object_id) + " has invalid stats object");
    }

    // Check that the agent has a valid reward pointer
    if (actor->reward == nullptr) {
      throw std::runtime_error("Agent with ID " + std::to_string(actor_object_id) + " has null reward pointer");
    }

    // Check if agent is frozen
    if (actor->frozen > 0) {
      actor->stats.incr("status.frozen.ticks");
      actor->stats.incr("status.frozen.ticks", actor->group_name);
      actor->frozen -= 1;
      return false;
    }

    // Call the derived implementation
    bool result = _handle_action(actor_id, actor, arg);

    debug_log << "1" << std::endl;

    // Update stats based on result
    if (result) {
      actor->stats.incr(_stats.success);
    } else {
      actor->stats.incr(_stats.failure);
      actor->stats.incr("action.failure_penalty");

      // Apply reward penalty - only check for nullptr
      if (actor->reward != nullptr) {
        *actor->reward -= actor->action_failure_penalty;
      }
    }
    debug_log << "3" << std::endl;
    // Set first_use stat
    actor->stats.set_once(_stats.first_use, current_timestep);

    return result;
  }

  virtual uint8_t max_arg() const {
    return 0;
  }

  std::string action_name() const {
    return _action_name;
  }

  // Clone method for ownership transfer - derived classes must implement
  virtual ActionHandler* clone() const = 0;

protected:
  // Pure virtual method to be implemented by derived classes
  virtual bool _handle_action(uint32_t actor_id, Agent* actor, c_actions_type arg) = 0;

  // Utility methods for derived classes
  void validate_orientation(Agent* actor) const {
    if (actor->orientation < 0 || actor->orientation > 3) {
      throw std::runtime_error("Invalid orientation " + std::to_string(actor->orientation) + " for agent " +
                               std::to_string(actor->id));
    }
  }

  bool is_valid_location(const GridLocation& loc) const {
    if (_grid == nullptr) {
      throw std::runtime_error("Grid not initialized when checking location validity");
    }
    return (loc.r < _grid->height && loc.c < _grid->width);
  }

  GridObject* safe_object_at(const GridLocation& loc) const {
    if (_grid == nullptr) {
      throw std::runtime_error("Grid not initialized when accessing object");
    }

    if (!is_valid_location(loc)) {
      return nullptr;  // Return nullptr for out-of-bounds locations
    }

    return _grid->object_at(loc);
  }

  StatNames _stats;
  std::string _action_name;  // Fixed variable name
};

#endif  // ACTION_HANDLER_HPP