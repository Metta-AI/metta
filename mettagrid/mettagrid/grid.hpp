#ifndef GRID_HPP
#define GRID_HPP

#include <algorithm>
#include <memory>
#include <vector>

#include "grid_object.hpp"

using namespace std;
typedef vector<vector<GridObjectId>> GridType;

class Grid {
public:
  unsigned int width;
  unsigned int height;

  GridType grid;
  vector<std::unique_ptr<GridObject>> objects;

  inline Grid(unsigned int width, unsigned int height) : width(width), height(height) {
    grid.resize(height, vector<GridObjectId>(width, 0));

    // 0 is reserved for empty space
    objects.push_back(nullptr);
  }

  virtual ~Grid() = default;

  inline char add_object(GridObject* obj) {
    if (obj->location.r >= height or obj->location.c >= width) {
      return false;
    }
    if (this->grid[obj->location.r][obj->location.c] != 0) {
      return false;
    }

    obj->id = this->objects.size();
    this->objects.push_back(std::unique_ptr<GridObject>(obj));
    this->grid[obj->location.r][obj->location.c] = obj->id;
    return true;
  }

  // Removes and object from the grid and gives ownership of the object to the caller.
  // Since the caller is now the owner, this can make the raw pointer invalid, if the
  // returned unique_ptr is destroyed.
  inline unique_ptr<GridObject> remove_object(GridObject* obj) {
    this->grid[obj->location.r][obj->location.c] = 0;
    auto obj_ptr = this->objects[obj->id].release();
    this->objects[obj->id] = nullptr;
    return std::unique_ptr<GridObject>(obj_ptr);
  }

  inline char move_object(GridObjectId id, const GridLocation& loc) {
    if (loc.r >= height or loc.c >= width) {
      return false;
    }

    if (grid[loc.r][loc.c] != 0) {
      return false;
    }

    GridObject* obj = object(id);
    grid[loc.r][loc.c] = id;
    grid[obj->location.r][obj->location.c] = 0;
    obj->location = loc;
    return true;
  }

  inline void swap_objects(GridObjectId id1, GridObjectId id2) {
    GridObject* obj1 = object(id1);
    GridLocation loc1 = obj1->location;
    grid[loc1.r][loc1.c] = 0;

    GridObject* obj2 = object(id2);
    GridLocation loc2 = obj2->location;
    grid[loc2.r][loc2.c] = 0;

    obj1->location = loc2;
    obj2->location = loc1;

    grid[obj1->location.r][obj1->location.c] = id1;
    grid[obj2->location.r][obj2->location.c] = id2;
  }

  inline GridObject* object(GridObjectId obj_id) {
    return objects[obj_id].get();
  }

  inline GridObject* object_at(const GridLocation& loc) {
    if (loc.r >= height or loc.c >= width) {
      return nullptr;
    }
    if (grid[loc.r][loc.c] == 0) {
      return nullptr;
    }
    return object(grid[loc.r][loc.c]);
  }

  inline GridObject* object_at(const GridLocation& loc, TypeId type_id) {
    GridObject* obj = object_at(loc);
    if (obj != NULL and obj->_type_id == type_id) {
      return obj;
    }
    return nullptr;
  }

  inline GridObject* object_at(GridCoord r, GridCoord c, TypeId type_id) {
    GridObject* obj = object_at(GridLocation(r, c));
    if (obj->_type_id != type_id) {
      return nullptr;
    }

    return obj;
  }

  inline const GridLocation location(GridObjectId id) {
    return object(id)->location;
  }

  inline const GridLocation relative_location(const GridLocation& loc,
                                              Orientation orientation,
                                              GridCoord distance,
                                              GridCoord offset) {
    int new_r = loc.r;
    int new_c = loc.c;

    switch (orientation) {
      case Up:
        new_r = loc.r - distance;
        new_c = loc.c - offset;
        break;
      case Down:
        new_r = loc.r + distance;
        new_c = loc.c + offset;
        break;
      case Left:
        new_r = loc.r + offset;
        new_c = loc.c - distance;
        break;
      case Right:
        new_r = loc.r - offset;
        new_c = loc.c + distance;
        break;
    }
    new_r = max(0, new_r);
    new_c = max(0, new_c);
    return GridLocation(new_r, new_c);
  }

  inline const GridLocation relative_location(const GridLocation& loc, Orientation orientation) {
    return this->relative_location(loc, orientation, 1, 0);
  }

  inline char is_empty(unsigned int row, unsigned int col) {
    GridLocation loc(row, col);
    if (object_at(loc) != nullptr) {
      return 0;
    }
    return 1;
  }
};

#endif  // GRID_HPP
