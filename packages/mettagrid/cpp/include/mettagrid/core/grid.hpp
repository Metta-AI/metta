#ifndef PACKAGES_METTAGRID_CPP_INCLUDE_METTAGRID_CORE_GRID_HPP_
#define PACKAGES_METTAGRID_CPP_INCLUDE_METTAGRID_CORE_GRID_HPP_

#include <algorithm>
#include <memory>
#include <vector>

#include "actions/orientation.hpp"
#include "core/grid_object.hpp"
#include "objects/constants.hpp"

using std::max;
using std::unique_ptr;
using std::vector;
using GridType = std::vector<std::vector<std::vector<GridObject*>>>;

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
    grid.resize(
        height,
        std::vector<std::vector<GridObject*>>(width, std::vector<GridObject*>(GridLayer::GridLayerCount, nullptr)));

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
    // GridCoord and Layer are unsigned; any negative input coerces to a large
    // value and fails these upper-bound checks, so explicit lower-bound checks
    // are not required for correctness.
    return loc.r < height && loc.c < width && loc.layer < GridLayer::GridLayerCount;
  }

  inline bool add_object(GridObject* obj) {
    if (!is_valid_location(obj->location)) {
      return false;
    }
    if (this->grid[obj->location.r][obj->location.c][obj->location.layer] != nullptr) {
      return false;
    }

    obj->id = static_cast<GridObjectId>(this->objects.size());
    this->objects.push_back(std::unique_ptr<GridObject>(obj));
    this->grid[obj->location.r][obj->location.c][obj->location.layer] = obj;
    return true;
  }

  inline bool move_object(GridObject& obj, const GridLocation& loc) {
    if (!is_valid_location(loc)) {
      return false;
    }

    // Disallow moving multi-cell objects to avoid leaving stale occupied cells.
    if (!obj.extra_cells.empty()) {
      return false;
    }

    if (grid[loc.r][loc.c][loc.layer] != nullptr) {
      return false;
    }

    grid[loc.r][loc.c][loc.layer] = &obj;
    grid[obj.location.r][obj.location.c][obj.location.layer] = nullptr;
    obj.location = loc;
    return true;
  }

  inline bool swap_objects(GridObject& obj1, GridObject& obj2) {
    // Reject swap if either object occupies multiple cells. Swapping only anchors would corrupt occupancy.
    if (!obj1.extra_cells.empty() || !obj2.extra_cells.empty()) {
      return false;
    }
    // Store the original locations.
    GridLocation loc1 = obj1.location;
    GridLocation loc2 = obj2.location;

    // Clear the objects from their original positions in the grid.
    grid[loc1.r][loc1.c][loc1.layer] = nullptr;
    grid[loc2.r][loc2.c][loc2.layer] = nullptr;

    // Update the location property of each object, preserving their original layers.
    obj1.location = {loc2.r, loc2.c, loc1.layer};
    obj2.location = {loc1.r, loc1.c, loc2.layer};

    // Place the objects in their new positions in the grid.
    grid[obj1.location.r][obj1.location.c][obj1.location.layer] = &obj1;
    grid[obj2.location.r][obj2.location.c][obj2.location.layer] = &obj2;
    return true;
  }

  // Helper to check if a location is free and valid for occupation on the same layer as obj
  inline bool can_occupy(const GridObject& obj, const GridLocation& loc) const {
    if (!is_valid_location(loc)) {
      return false;
    }
    if (loc.layer != obj.location.layer) {
      return false;  // single-layer only
    }
    if (grid[loc.r][loc.c][loc.layer] != nullptr) {
      return false;
    }
    return true;
  }

  // Get full set of occupied cells for an object including anchor.
  inline std::vector<GridLocation> occupied_cells(const GridObject& obj) const {
    std::vector<GridLocation> cells;
    cells.reserve(1 + obj.extra_cells.size());
    cells.push_back(obj.location);
    for (const auto& loc : obj.extra_cells) {
      cells.push_back(loc);
    }
    return cells;
  }

  // Check 4-neighbor connectivity for a set of cells; assumes all same layer
  inline bool is_connected_shape(const std::vector<GridLocation>& cells) const {
    if (cells.empty()) return true;
    // Simple BFS over indices using 4-neighborhood
    auto neighbors4 = [](const GridLocation& a, const GridLocation& b) {
      return a.layer == b.layer &&
             ((a.r == b.r && (a.c == b.c + 1 || a.c + 1 == b.c)) ||
              (a.c == b.c && (a.r == b.r + 1 || a.r + 1 == b.r)));
    };
    std::vector<char> visited(cells.size(), 0);
    visited[0] = 1;
    std::vector<size_t> queue;
    queue.push_back(0);
    size_t qi = 0;
    while (qi < queue.size()) {
      size_t i = queue[qi++];
      for (size_t j = 0; j < cells.size(); j++) {
        if (!visited[j] && neighbors4(cells[i], cells[j])) {
          visited[j] = 1;
          queue.push_back(j);
        }
      }
    }
    for (char v : visited) {
      if (!v) return false;
    }
    return true;
  }

  // Occupy a new cell for an existing object. Returns true on success.
  // Returns false if: location invalid/occupied/wrong layer or shape not connected.
  inline bool occupy_cell(GridObject& obj, const GridLocation& loc) {
    if (!can_occupy(obj, loc)) return false;
    // Tentatively add and check connectivity (anchor + extra + new loc)
    auto cells = occupied_cells(obj);
    cells.push_back(loc);
    if (!is_connected_shape(cells)) {
      return false;
    }
    grid[loc.r][loc.c][loc.layer] = &obj;
    obj.extra_cells.push_back(loc);
    return true;
  }

  // Release a non-anchor occupied cell. Returns true on success.
  inline bool release_cell(GridObject& obj, const GridLocation& loc) {
    // Cannot release anchor
    if (loc.r == obj.location.r && loc.c == obj.location.c && loc.layer == obj.location.layer) {
      return false;
    }
    // Find in extra_cells
    auto it = std::find_if(obj.extra_cells.begin(), obj.extra_cells.end(), [&](const GridLocation& l) {
      return l.r == loc.r && l.c == loc.c && l.layer == loc.layer;
    });
    if (it == obj.extra_cells.end()) return false;
    // Tentatively remove and check connectivity
    auto cells = occupied_cells(obj);
    // remove one matching entry from cells vector
    auto it2 = std::find_if(cells.begin(), cells.end(), [&](const GridLocation& l) {
      return l.r == loc.r && l.c == loc.c && l.layer == loc.layer;
    });
    if (it2 != cells.end()) {
      cells.erase(it2);
    }
    if (!is_connected_shape(cells)) {
      return false;
    }
    // Apply removal
    grid[loc.r][loc.c][loc.layer] = nullptr;
    obj.extra_cells.erase(it);
    return true;
  }

  // Removed unused batch helpers occupy_cells/release_cells to reduce surface area.

  inline GridObject* object(GridObjectId obj_id) const {
    assert(obj_id < objects.size() && "Invalid object ID");
    return objects[obj_id].get();
  }

  inline GridObject* object_at(const GridLocation& loc) const {
    if (!is_valid_location(loc)) {
      return nullptr;
    }
    return grid[loc.r][loc.c][loc.layer];
  }

  inline const GridLocation relative_location(const GridLocation& loc,
                                              Orientation facing,
                                              short forward_distance,
                                              short lateral_offset) {
    const int r = static_cast<int>(loc.r);
    const int c = static_cast<int>(loc.c);
    int new_r = r;
    int new_c = c;

    // Get the forward direction deltas
    int forward_dr, forward_dc;
    getOrientationDelta(facing, forward_dc, forward_dr);

    // Apply forward/backward movement
    new_r += forward_dr * forward_distance;
    new_c += forward_dc * forward_distance;

    // Apply lateral movement (right/left)
    if (lateral_offset != 0) {
      // Right is 90 degrees clockwise from facing direction
      Orientation right_facing = getClockwise(facing);

      // Get the right direction deltas
      int right_dr, right_dc;
      getOrientationDelta(right_facing, right_dc, right_dr);

      // Apply lateral movement
      new_r += right_dr * lateral_offset;
      new_c += right_dc * lateral_offset;
    }

    // Clamp to grid bounds
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
      if (layer_objects != nullptr) return false;
    }
    return true;
  }

  // is_empty for a specific layer
  inline bool is_empty_at_layer(GridCoord row, GridCoord col, ObservationType layer) const {
    if (grid[row][col][layer] != nullptr) return false;
    return true;
  }
};

#endif  // PACKAGES_METTAGRID_CPP_INCLUDE_METTAGRID_CORE_GRID_HPP_
