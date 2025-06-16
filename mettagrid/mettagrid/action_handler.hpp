#ifndef METTAGRID_METTAGRID_ACTION_HANDLER_HPP_
#define METTAGRID_METTAGRID_ACTION_HANDLER_HPP_

#include <map>
#include <string>
#include <vector>

#include "grid.hpp"
#include "grid_object.hpp"
#include "objects/agent.hpp"
#include "objects/constants.hpp"
#include "types.hpp"

typedef std::map<std::string, int> ActionConfig;

class ActionHandler {
public:
  unsigned char priority;
  Grid* _grid;

  ActionHandler(const ActionConfig& cfg, const std::string& action_name) : priority(0), _action_name(action_name) {}

  virtual ~ActionHandler() {}

  void init(Grid* grid) {
    this->_grid = grid;
  }

  bool handle_action(GridObjectId actor_object_id, ActionArg arg) {
    Agent* actor = static_cast<Agent*>(_grid->object(actor_object_id));

    // Handle frozen status
    if (actor->frozen > 0) {
      actor->stats.incr("status.frozen.ticks");
      actor->stats.incr("status.frozen.ticks." + actor->group_name);
      actor->frozen -= 1;
      return false;
    }

    // Execute the action
    bool result = _handle_action(actor, arg);

    // Track success/failure
    if (result) {
      actor->stats.incr("action." + _action_name + ".success");
    } else {
      actor->stats.incr("action." + _action_name + ".failed");
      actor->stats.incr("action.failure_penalty");
      *actor->reward -= actor->action_failure_penalty;
    }

    return result;
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
};

#endif  // METTAGRID_METTAGRID_ACTION_HANDLER_HPP_
