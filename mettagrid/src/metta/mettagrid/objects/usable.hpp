#ifndef OBJECTS_USABLE_HPP_
#define OBJECTS_USABLE_HPP_

#include "grid_object.hpp"
#include "types.hpp"

// Forward declaration
class Agent;

class Usable : public GridObject {
public:
  virtual ~Usable() = default;

  virtual bool onUse(Agent* actor, ActionArg arg) = 0;
};

#endif  // OBJECTS_USABLE_HPP_
