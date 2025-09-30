#ifndef PACKAGES_METTAGRID_CPP_INCLUDE_METTAGRID_OBJECTS_CLIPPED_GRID_OBJECT_HPP_
#define PACKAGES_METTAGRID_CPP_INCLUDE_METTAGRID_OBJECTS_CLIPPED_GRID_OBJECT_HPP_

#include <map>
#include <stdexcept>
#include <string>
#include <vector>

#include "core/grid.hpp"
#include "core/grid_object.hpp"
#include "core/types.hpp"
#include "objects/assembler.hpp"
#include "objects/clipper.hpp"
#include "objects/constants.hpp"

// Represents a Clipped GridObject. This is a more-or-less imprisoned GridObject
// that can be freed by successfully using an Assembler Recipe.

class ClippedGridObject : public Assembler {
public:
  Clipper* clipper;
  // When an object is clipped, it's removed from the grid and held here.
  std::unique_ptr<GridObject> clipped_object;
  ClippedGridObject(GridCoord r,
                    GridCoord c,
                    const ClippedGridObjectConfig& cfg,
                    Clipper* clipper,
                    std::unique_ptr<GridObject> clipped_object)
      : Assembler(r, c, cfg), clipped_object(std::move(clipped_object)) {}

  // Implement pure virtual method from Usable
  virtual bool onUse(Agent& actor, ActionArg arg) override {
    if (Assembler::onUse(actor, arg)) {
      clipper->unclip(this);
      return true;
    }
    return false;
  }
};

#endif  // PACKAGES_METTAGRID_CPP_INCLUDE_METTAGRID_OBJECTS_CLIPPED_GRID_OBJECT_HPP_
