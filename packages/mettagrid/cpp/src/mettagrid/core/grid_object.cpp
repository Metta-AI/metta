#include "core/grid_object.hpp"

#include "handler/handler.hpp"
#include "handler/handler_context.hpp"
#include "objects/agent.hpp"

bool GridObject::onUse(Agent& actor, ActionArg /*arg*/) {
  // Try each on_use handler in order until one succeeds
  for (auto& handler : _on_use_handlers) {
    if (handler->try_apply(&actor, this)) {
      return true;
    }
  }
  return false;
}

void GridObject::fire_on_update_handlers() {
  // For on_update handlers, actor is nullptr and target is this object
  // Set skip_on_update_trigger=true to prevent infinite recursion
  mettagrid::HandlerContext ctx(nullptr, this, /*skip_on_update_trigger=*/true);

  // Try each on_update handler - all that pass filters will be applied
  for (auto& handler : _on_update_handlers) {
    handler->try_apply(ctx);
  }
}
