#include "core/grid_object.hpp"

#include "handler/handler.hpp"
#include "objects/agent.hpp"

bool GridObject::onUse(Agent& actor, ActionArg /*arg*/) {
  // Try each handler in order until one succeeds
  for (auto& handler : _handlers) {
    if (handler->try_apply(&actor, this)) {
      return true;
    }
  }
  return false;
}
