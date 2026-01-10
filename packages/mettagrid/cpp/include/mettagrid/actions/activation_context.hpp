#ifndef PACKAGES_METTAGRID_CPP_INCLUDE_METTAGRID_ACTIONS_ACTIVATION_CONTEXT_HPP_
#define PACKAGES_METTAGRID_CPP_INCLUDE_METTAGRID_ACTIONS_ACTIVATION_CONTEXT_HPP_

#include "actions/activation_handler_config.hpp"
#include "core/grid_object.hpp"
#include "objects/collective.hpp"
#include "objects/has_inventory.hpp"

namespace mettagrid {

/**
 * ActivationContext holds references to all entities involved in an activation
 * and provides entity resolution for filters and mutations.
 */
class ActivationContext {
public:
  HasInventory* actor = nullptr;
  HasInventory* target = nullptr;

  ActivationContext() = default;
  ActivationContext(HasInventory* act, HasInventory* tgt) : actor(act), target(tgt) {}

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

  // Get the collective for an entity (if it's a GridObject)
  Collective* get_collective(HasInventory* entity) const {
    if (entity == nullptr) {
      return nullptr;
    }
    // All GridObjects are Alignable, so try to cast to GridObject
    GridObject* grid_obj = dynamic_cast<GridObject*>(entity);
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
};

}  // namespace mettagrid

#endif  // PACKAGES_METTAGRID_CPP_INCLUDE_METTAGRID_ACTIONS_ACTIVATION_CONTEXT_HPP_
