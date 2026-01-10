#ifndef PACKAGES_METTAGRID_CPP_INCLUDE_METTAGRID_OBJECTS_USABLE_HPP_
#define PACKAGES_METTAGRID_CPP_INCLUDE_METTAGRID_OBJECTS_USABLE_HPP_

#include "core/types.hpp"

// Forward declaration
class Agent;

class Usable {
public:
  virtual ~Usable() = default;

  // Default implementation returns false (no-op)
  virtual bool onUse(Agent& actor, ActionArg arg) {
    (void)actor;
    (void)arg;
    return false;
  }
};

#endif  // PACKAGES_METTAGRID_CPP_INCLUDE_METTAGRID_OBJECTS_USABLE_HPP_
