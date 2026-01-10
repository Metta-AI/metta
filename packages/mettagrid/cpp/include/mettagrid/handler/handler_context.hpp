#ifndef PACKAGES_METTAGRID_CPP_INCLUDE_METTAGRID_HANDLER_HANDLER_CONTEXT_HPP_
#define PACKAGES_METTAGRID_CPP_INCLUDE_METTAGRID_HANDLER_HANDLER_CONTEXT_HPP_

#include "handler/handler_config.hpp"
#include "objects/alignable.hpp"
#include "objects/collective.hpp"
#include "objects/has_inventory.hpp"

namespace mettagrid {

/**
 * HandlerContext holds references to all entities involved in an activation
 * and provides entity resolution for filters and mutations.
 */
class HandlerContext {
public:
  HasInventory* actor = nullptr;
  HasInventory* target = nullptr;

  HandlerContext() = default;
  HandlerContext(HasInventory* act, HasInventory* tgt) : actor(act), target(tgt) {}

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

  // Get the collective for an entity (if alignable)
  Collective* get_collective(HasInventory* entity) const {
    Alignable* alignable = dynamic_cast<Alignable*>(entity);
    if (alignable == nullptr) {
      return nullptr;
    }
    return alignable->getCollective();
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

#endif  // PACKAGES_METTAGRID_CPP_INCLUDE_METTAGRID_HANDLER_HANDLER_CONTEXT_HPP_
