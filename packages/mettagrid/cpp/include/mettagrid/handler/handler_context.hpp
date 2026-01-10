#ifndef PACKAGES_METTAGRID_CPP_INCLUDE_METTAGRID_HANDLER_HANDLER_CONTEXT_HPP_
#define PACKAGES_METTAGRID_CPP_INCLUDE_METTAGRID_HANDLER_HANDLER_CONTEXT_HPP_

#include "core/grid_object.hpp"
#include "handler/handler_config.hpp"
#include "objects/collective.hpp"
#include "objects/has_inventory.hpp"

namespace mettagrid {

/**
 * HandlerContext holds references to all entities involved in a handler execution
 * and provides entity resolution for filters and mutations.
 *
 * Context varies by handler type:
 *   - on_use: actor=agent performing action, target=object being used
 *   - on_update: actor=nullptr, target=object being updated
 *   - aoe: actor=source object, target=affected object
 */
class HandlerContext {
public:
  HasInventory* actor = nullptr;
  HasInventory* target = nullptr;

  // Flag to prevent infinite recursion when on_update handlers trigger more mutations
  bool skip_on_update_trigger = false;

  HandlerContext() = default;
  HandlerContext(HasInventory* act, HasInventory* tgt, bool skip_update = false)
      : actor(act), target(tgt), skip_on_update_trigger(skip_update) {}

  // Resolve an EntityRef to the corresponding HasInventory*
  HasInventory* resolve(EntityRef ref) const {
    switch (ref) {
      case EntityRef::actor:
        return actor;
      case EntityRef::target:
        return target;
      case EntityRef::actor_collective:
        return get_collective(actor);
      case EntityRef::target_collective:
        return get_collective(target);
      default:
        return nullptr;
    }
  }

  // Get GridObject for an entity (if it's a GridObject)
  GridObject* get_grid_object(HasInventory* entity) const {
    if (entity == nullptr) {
      return nullptr;
    }
    return dynamic_cast<GridObject*>(entity);
  }

  // Get the collective for an entity (if it's a GridObject)
  Collective* get_collective(HasInventory* entity) const {
    GridObject* grid_obj = get_grid_object(entity);
    if (grid_obj == nullptr) {
      return nullptr;
    }
    return grid_obj->getCollective();
  }

  // Get actor's collective
  Collective* actor_collective() const {
    return get_collective(actor);
  }

  // Get target's collective
  Collective* target_collective() const {
    return get_collective(target);
  }

  // Get actor as GridObject (for vibe access, etc.)
  GridObject* actor_grid_object() const {
    return get_grid_object(actor);
  }

  // Get target as GridObject (for vibe access, etc.)
  GridObject* target_grid_object() const {
    return get_grid_object(target);
  }

  // Get actor's vibe (returns 0 if actor is not a GridObject)
  ObservationType actor_vibe() const {
    GridObject* grid_obj = actor_grid_object();
    return grid_obj != nullptr ? grid_obj->vibe : 0;
  }

  // Get target's vibe (returns 0 if target is not a GridObject)
  ObservationType target_vibe() const {
    GridObject* grid_obj = target_grid_object();
    return grid_obj != nullptr ? grid_obj->vibe : 0;
  }
};

}  // namespace mettagrid

#endif  // PACKAGES_METTAGRID_CPP_INCLUDE_METTAGRID_HANDLER_HANDLER_CONTEXT_HPP_
