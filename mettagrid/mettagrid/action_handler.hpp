#ifndef ACTION_HANDLER_HPP
#define ACTION_HANDLER_HPP

#include <map>
#include <string>
#include <vector>

#include "grid.hpp"
#include "grid_object.hpp"
#include "objects/agent.hpp"
#include "objects/constants.hpp"

struct StatNames {
  std::string success;
  std::string first_use;
  std::string failure;

  std::map<TypeId, std::string> target;
  std::map<TypeId, std::string> target_first_use;
  std::vector<std::string> group;
};

typedef unsigned char ActionArg;
typedef std::map<std::string, int> ActionConfig;

class ActionHandler {
public:
  unsigned char priority;
  Grid* _grid;

  ActionHandler(const ActionConfig& cfg, const std::string& action_name) : priority(0), _action_name(action_name) {
    _stats.success = "action." + action_name;
    _stats.failure = "action." + action_name + ".failed";
    _stats.first_use = "action." + action_name + ".first_use";

    for (TypeId t = 0; t < ObjectType::Count; t++) {
      _stats.target[t] = _stats.success + "." + ObjectTypeNames[t];
      _stats.target_first_use[t] = _stats.first_use + "." + ObjectTypeNames[t];
    }
  }

  void init(Grid* grid) {
    this->_grid = grid;
  }

  bool handle_action(unsigned int actor_id,
                     GridObjectId actor_object_id,
                     ActionArg arg,
                     unsigned int current_timestep) {
    Agent* actor = static_cast<Agent*>(_grid->object(actor_object_id));

    if (actor->frozen > 0) {
      actor->stats.incr("status.frozen.ticks");
      actor->stats.incr("status.frozen.ticks", actor->group_name);
      actor->frozen -= 1;
      return false;
    }

    bool result = _handle_action(actor_id, actor, arg);

    if (result) {
      actor->stats.incr(_stats.success);
    } else {
      actor->stats.incr(_stats.failure);
      actor->stats.incr("action.failure_penalty");
      *actor->reward -= actor->action_failure_penalty;
      actor->stats.set_once(_stats.first_use, current_timestep);
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
  virtual bool _handle_action(unsigned int actor_id, Agent* actor, ActionArg arg) = 0;

  StatNames _stats;
  std::string _action_name;
};

#endif  // ACTION_HANDLER_HPP
