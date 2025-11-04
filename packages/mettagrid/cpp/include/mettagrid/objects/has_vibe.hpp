#ifndef PACKAGES_METTAGRID_CPP_INCLUDE_METTAGRID_OBJECTS_HAS_VIBE_HPP_
#define PACKAGES_METTAGRID_CPP_INCLUDE_METTAGRID_OBJECTS_HAS_VIBE_HPP_

#include "core/types.hpp"
#include "objects/constants.hpp"

class HasVibe {
public:
  ObservationType vibe = 0;

  explicit HasVibe(ObservationType initial_vibe = 0) : vibe(initial_vibe) {}

  virtual ~HasVibe() = default;

  virtual void set_vibe(ObservationType new_vibe) {
    vibe = new_vibe;
  }
};

#endif  // PACKAGES_METTAGRID_CPP_INCLUDE_METTAGRID_OBJECTS_HAS_VIBE_HPP_
