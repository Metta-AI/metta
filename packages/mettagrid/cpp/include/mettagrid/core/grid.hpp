#ifndef PACKAGES_METTAGRID_CPP_INCLUDE_METTAGRID_CORE_GRID_HPP_
#define PACKAGES_METTAGRID_CPP_INCLUDE_METTAGRID_CORE_GRID_HPP_

#include <algorithm>
#include <memory>
#include <vector>

#include "core/grid_object.hpp"
#include "objects/constants.hpp"

using std::max;
using std::unique_ptr;
using std::vector;
using GridType = std::vector<std::vector<GridObject*>>;

class Grid {
public:
  const GridCoord height;
  const GridCoord width;
  vector<std::unique_ptr<GridObject>> objects;

private:
  GridType grid;

public:
  Grid(GridCoord height, GridCoord width)
      : height(height),
        width(width),
        objects(),  // Initialize objects in member init list
        grid() {    // Initialize grid in member init list
    grid.resize(height, std::vector<GridObject*>(width, nullptr));

    // Reserve space for objects to avoid frequent reallocations
    // Assume ~50% of grid cells will contain objects
    size_t estimated_objects = static_cast<size_t>(height) * width / 2;

    // Cap preallocation at ~100MB of pointer memory
    constexpr size_t MAX_PREALLOCATED_OBJECTS = 12'500'000;
    size_t reserved_objects = std::min(estimated_objects, MAX_PREALLOCATED_OBJECTS);

    objects.reserve(reserved_objects);

    // GridObjectId "0" is reserved to mean empty space (GridObject pointer = nullptr).
    // By pushing nullptr at index 0, we ensure that:
    //   1. Grid initialization with zeros automatically represents empty spaces
    //   2. Object IDs match their index in the objects vector (no off-by-one adjustments)
    objects.push_back(nullptr);
  }

  ~Grid() = default;

  inline bool is_valid_location(const GridLocation& loc) const {
    return loc.r < height && loc.c < width;
  }

  inline bool add_object(GridObject* obj) {
    if (!is_valid_location(obj->location)) {
      return false;
    }
    if (this->grid[obj->location.r][obj->location.c] != nullptr) {
      return false;
    }

    obj->id = static_cast<GridObjectId>(this->objects.size());
    this->objects.push_back(std::unique_ptr<GridObject>(obj));
    this->grid[obj->location.r][obj->location.c] = obj;
    return true;
  }

  inline bool move_object(GridObject& obj, const GridLocation& loc) {
    if (!is_valid_location(loc)) {
      return false;
    }

    if (grid[loc.r][loc.c] != nullptr) {
      return false;
    }

    grid[loc.r][loc.c] = &obj;
    grid[obj.location.r][obj.location.c] = nullptr;
    obj.location = loc;
    return true;
  }

  inline bool swap_objects(GridObject& obj1, GridObject& obj2) {
    GridLocation loc1 = obj1.location;
    GridLocation loc2 = obj2.location;

    grid[loc1.r][loc1.c] = &obj2;
    grid[loc2.r][loc2.c] = &obj1;
    obj1.location = loc2;
    obj2.location = loc1;
    return true;
  }

  inline GridObject* object(GridObjectId obj_id) const {
    assert(obj_id < objects.size() && "Invalid object ID");
    return objects[obj_id].get();
  }

  inline GridObject* object_at(const GridLocation& loc) const {
    if (!is_valid_location(loc)) {
      return nullptr;
    }
    return grid[loc.r][loc.c];
  }

  inline bool is_empty(GridCoord row, GridCoord col) const {
    return grid[row][col] == nullptr;
  }
};

#endif  // PACKAGES_METTAGRID_CPP_INCLUDE_METTAGRID_CORE_GRID_HPP_
