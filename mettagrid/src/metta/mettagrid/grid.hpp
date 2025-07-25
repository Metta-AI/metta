#ifndef GRID_HPP_
#define GRID_HPP_

#include <algorithm>
#include <memory>
#include <vector>
#include <set>

#include "grid_object.hpp"
#include "objects/constants.hpp"

using std::max;
using std::unique_ptr;
using std::vector;
using GridType = std::vector<std::vector<std::vector<GridObjectId>>>;

class Grid {
public:
  const GridCoord height;
  const GridCoord width;
  vector<std::unique_ptr<GridObject>> objects;
  std::set<size_t> null_object_indices; // Indices in objects vector that are nullptr

private:
  GridType grid;

public:
  Grid(GridCoord height, GridCoord width) : height(height), width(width) {
    grid.resize(height,
                std::vector<std::vector<GridObjectId>>(width, std::vector<GridObjectId>(GridLayer::GridLayerCount, 0)));

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
    return loc.r < height && loc.c < width && loc.layer < GridLayer::GridLayerCount;
  }

  inline bool add_object(GridObject* obj) {
    // If there are any null slots, reuse the lowest one
    if (!null_object_indices.empty()) {
      size_t idx = *null_object_indices.begin();
      null_object_indices.erase(null_object_indices.begin());
      obj->id = static_cast<GridObjectId>(idx);
      objects[idx].reset(obj);
      grid[obj->location.r][obj->location.c][obj->location.layer] = obj->id;
      return true;
    }
    // Otherwise, append to the end
    obj->id = static_cast<GridObjectId>(objects.size());
    objects.push_back(std::unique_ptr<GridObject>(obj));
    grid[obj->location.r][obj->location.c][obj->location.layer] = obj->id;
    return true;
  }

  // Removes an object from the grid and gives ownership of the object to the caller.
  // Since the caller is now the owner, this can make the raw pointer invalid, if the
  // returned unique_ptr is destroyed.
  inline std::unique_ptr<GridObject> remove_object(GridObject* obj) {
    grid[obj->location.r][obj->location.c][obj->location.layer] = 0;
    auto obj_ptr = objects[obj->id].release();
    objects[obj->id] = nullptr;
    null_object_indices.insert(obj->id);
    return std::unique_ptr<GridObject>(obj_ptr);
  }


  inline bool move_object(GridObjectId id, const GridLocation& loc) {
    if (!is_valid_location(loc)) {
      return false;
    }

    if (grid[loc.r][loc.c][loc.layer] != 0) {
      return false;
    }

    GridObject* obj = object(id);
    grid[loc.r][loc.c][loc.layer] = id;
    grid[obj->location.r][obj->location.c][obj->location.layer] = 0;
    obj->location = loc;
    return true;
  }

  inline void swap_objects(GridObjectId id1, GridObjectId id2) {
    GridObject* obj1 = object(id1);
    GridObject* obj2 = object(id2);

    // Store the original locations.
    GridLocation loc1 = obj1->location;
    GridLocation loc2 = obj2->location;

    // Clear the objects from their original positions in the grid.
    grid[loc1.r][loc1.c][loc1.layer] = 0;
    grid[loc2.r][loc2.c][loc2.layer] = 0;

    // Update the location property of each object, preserving their original layers.
    obj1->location = {loc2.r, loc2.c, loc1.layer};
    obj2->location = {loc1.r, loc1.c, loc2.layer};

    // Place the objects in their new positions in the grid.
    grid[obj1->location.r][obj1->location.c][obj1->location.layer] = id1;
    grid[obj2->location.r][obj2->location.c][obj2->location.layer] = id2;
  }

  inline GridObject* object(GridObjectId obj_id) const {
    assert(obj_id < objects.size() && "Invalid object ID");
    return objects[obj_id].get();
  }

  inline GridObject* object_at(const GridLocation& loc) const {
    if (!is_valid_location(loc)) {
      return nullptr;
    }
    if (grid[loc.r][loc.c][loc.layer] == 0) {
      return nullptr;
    }
    return object(grid[loc.r][loc.c][loc.layer]);
  }

  /**
   * Get the location at a relative offset from the given orientation.
   * Note: The returned location has the same layer as the input location.
   */
  inline const GridLocation relative_location(const GridLocation& loc,
                                              Orientation facing,
                                              short forward_distance,  // + is forward, - is backward
                                              short lateral_offset) {  // + is relative right, - is relative left
    const int r = static_cast<int>(loc.r);
    const int c = static_cast<int>(loc.c);

    int new_r;
    int new_c;

    switch (facing) {
      case Orientation::Up:
        new_r = r - forward_distance;  // Positive dist = go up (decrease row)
        new_c = c + lateral_offset;    // Positive offset = go right (increase col)
        break;
      case Orientation::Down:
        new_r = r + forward_distance;  // Positive dist = go down (increase row)
        new_c = c - lateral_offset;    // Positive offset = go left (decrease col)
        break;
      case Orientation::Left:
        new_c = c - forward_distance;  // Positive dist = go left (decrease col)
        new_r = r - lateral_offset;    // Positive offset = go up (decrease row)
        break;
      case Orientation::Right:
        new_c = c + forward_distance;  // Positive dist = go right (increase col)
        new_r = r + lateral_offset;    // Positive offset = go down (increase row)
        break;
      default:
        assert(false && "Invalid orientation passed to relative_location()");
    }
    new_r = std::clamp(new_r, 0, static_cast<int>(this->height - 1));
    new_c = std::clamp(new_c, 0, static_cast<int>(this->width - 1));
    return GridLocation(static_cast<GridCoord>(new_r), static_cast<GridCoord>(new_c), loc.layer);
  }

  /**
   * Get the location one step forward in the given orientation.
   * Note: The returned location has the same layer as the input location.
   */
  inline const GridLocation relative_location(const GridLocation& loc, Orientation orientation) {
    return this->relative_location(loc, orientation, 1, 0);
  }

  inline bool is_empty(GridCoord row, GridCoord col) const {
    for (const auto& layer_objects : grid[row][col]) {
      if (layer_objects != 0) return false;
    }
    return true;
  }
};

#endif  // GRID_HPP_
