#ifndef PACKAGES_METTAGRID_CPP_INCLUDE_METTAGRID_OBJECTS_USABLE_HPP_
#define PACKAGES_METTAGRID_CPP_INCLUDE_METTAGRID_OBJECTS_USABLE_HPP_
#ifndef OBJECTS_USABLE_HPP_
#define OBJECTS_USABLE_HPP_

#include "core/grid_object.hpp"
#include "core/types.hpp"

// Forward declaration
class Agent;

class Usable {
public:
  virtual ~Usable() = default;

  virtual bool onUse(Agent& actor, ActionArg arg) = 0;
};

#endif  // OBJECTS_USABLE_HPP_

#endif  // PACKAGES_METTAGRID_CPP_INCLUDE_METTAGRID_OBJECTS_USABLE_HPP_
