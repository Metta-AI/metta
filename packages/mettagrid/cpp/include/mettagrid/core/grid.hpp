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
    return loc.r < height && loc.c < width && loc.layer < GridLayer::GridLayerCount;
  }

  inline bool add_object(GridObject* obj) {
    if (obj == nullptr) {
      return false;
    }
    if (obj->present_on_grid) {
      return false;
    }
    if (obj->locations.empty()) {
      return false;
    }
    obj->previous_locations = obj->locations;
    for (const auto& loc : obj->locations) {
      if (!is_valid_location(loc)) {
        return false;
      }
      if (grid[loc.r][loc.c][loc.layer] != nullptr) {
        return false;
      }
    }
    obj->id = static_cast<GridObjectId>(objects.size());
    objects.push_back(std::unique_ptr<GridObject>(obj));
    for (const auto& loc : obj->locations) {
      grid[loc.r][loc.c][loc.layer] = obj;
    }
    obj->present_on_grid = true;
    return true;
  }

  inline bool move_object(GridObject& obj, const GridLocation& loc) {
    if (!is_valid_location(loc)) {
      return false;
    }
    if (!obj.present_on_grid) {
      return false;
    }
    if (obj.locations.size() != 1) {
      return false;
    }
    if (grid[loc.r][loc.c][loc.layer] != nullptr) {
      return false;
    }

    GridLocation old = obj.locations[0];
    grid[old.r][old.c][old.layer] = nullptr;
    obj.locations[0] = loc;
    grid[loc.r][loc.c][loc.layer] = &obj;
    return true;
  }

  inline bool swap_objects(GridObject& obj1, GridObject& obj2) {
    if (!obj1.present_on_grid || !obj2.present_on_grid) {
      return false;
    }
    if (obj1.locations.size() != 1 || obj2.locations.size() != 1) {
      return false;
    }
    GridLocation loc1 = obj1.locations[0];
    GridLocation loc2 = obj2.locations[0];

    grid[loc1.r][loc1.c][loc1.layer] = nullptr;
    grid[loc2.r][loc2.c][loc2.layer] = nullptr;

    obj1.locations[0] = {loc2.r, loc2.c, loc1.layer};
    obj2.locations[0] = {loc1.r, loc1.c, loc2.layer};

    grid[obj1.locations[0].r][obj1.locations[0].c][obj1.locations[0].layer] = &obj1;
    grid[obj2.locations[0].r][obj2.locations[0].c][obj2.locations[0].layer] = &obj2;
    return true;
  }

  inline bool can_occupy(const GridObject& obj, const GridLocation& loc) const {
    if (!is_valid_location(loc)) {
      return false;
    }
    if (grid[loc.r][loc.c][loc.layer] != nullptr) {
      return false;
    }
    if (!obj.locations.empty() && obj.locations[0].layer != loc.layer) {
      return false;
    }
    return true;
  }

  // Get full set of occupied locations for an object.
  inline std::vector<GridLocation> occupied_locations(const GridObject& obj) const {
    if (!obj.present_on_grid) return {};
    return obj.locations;
  }

  // Occupy a new location for an existing object. Returns true on success.
  // Returns false if: location invalid/occupied or wrong layer.
  // If object is deactivated, this will reactivate it first.
  // Objects that don't support multi-cell will be rejected if already occupying a cell.
  inline bool occupy_location(GridObject& obj, const GridLocation& loc) {
    // Enforce single-cell for objects that don't support multi-cell
    if (!obj.supports_multi_cell() && obj.locations.size() >= 1) {
      return false;  // Cannot add cells to single-cell-only objects
    }

    // If object is deactivated, reactivate it first
    if (!obj.present_on_grid) {
      if (!activate_object(obj)) {
        return false;
      }
    }
    if (!can_occupy(obj, loc)) {
      return false;
    }
    grid[loc.r][loc.c][loc.layer] = &obj;
    obj.locations.push_back(loc);
    return true;
  }

  // Release an occupied location. Returns true on success.
  // Cannot remove the last cell; use deactivate_object() to go off-grid.
  // Cannot remove the anchor cell (locations[0]) to maintain invariant.
  inline bool release_location(GridObject& obj, const GridLocation& loc) {
    // Disallow removing the last cell
    if (obj.locations.size() <= 1) {
      return false;
    }
    // Find in current locations
    auto it = std::find_if(obj.locations.begin(), obj.locations.end(), [&](const GridLocation& l) {
      return l.r == loc.r && l.c == loc.c && l.layer == loc.layer;
    });
    if (it == obj.locations.end()) return false;

    // Disallow removing the anchor cell (locations[0])
    if (it == obj.locations.begin()) {
      return false;
    }

    // Apply removal in grid and object state
    if (is_valid_location(loc) && grid[loc.r][loc.c][loc.layer] == &obj) {
      grid[loc.r][loc.c][loc.layer] = nullptr;
    }
    obj.locations.erase(it);
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
    return grid[loc.r][loc.c][loc.layer];
  }

  // Deactivate an object: remove all occupancy; clears locations.
  inline bool deactivate_object(GridObject& obj) {
    for (const auto& loc : obj.locations) {
      if (is_valid_location(loc) && grid[loc.r][loc.c][loc.layer] == &obj) {
        grid[loc.r][loc.c][loc.layer] = nullptr;
      }
    }
    obj.previous_locations = obj.locations;
    obj.locations.clear();
    obj.present_on_grid = false;
    return true;
  }

  // Activate an object: if it has locations, place them; else restore from previous_locations
  inline bool activate_object(GridObject& obj) {
    if (obj.present_on_grid) return true;
    // Determine candidate locations to place
    const std::vector<GridLocation>* candidates = &obj.locations;
    if (obj.locations.empty()) {
      if (obj.previous_locations.empty()) return false;
      candidates = &obj.previous_locations;
    }
    // Validate candidates before placing
    for (const auto& loc : *candidates) {
      if (!is_valid_location(loc)) return false;
      if (grid[loc.r][loc.c][loc.layer] != nullptr) return false;
    }
    // Place all candidate locations
    obj.locations = *candidates;
    for (const auto& loc : obj.locations) {
      grid[loc.r][loc.c][loc.layer] = &obj;
    }
    obj.present_on_grid = true;
    return true;
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
