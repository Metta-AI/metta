#ifndef GRID_HPP_
#define GRID_HPP_

#include <algorithm>
#include <memory>
#include <vector>

#include "grid_object.hpp"
#include "objects/constants.hpp"

using std::max;
using std::unique_ptr;
using std::vector;
using GridType = std::vector<std::vector<std::vector<GridObjectId>>>;

class Grid {
public:
  unsigned int width;
  unsigned int height;

  GridType grid;
  vector<std::unique_ptr<GridObject>> objects;

  inline Grid(unsigned int width, unsigned int height) : width(width), height(height) {
    grid.resize(height, vector<vector<GridObjectId>>(width, vector<GridObjectId>(GridLayer::GridLayerCount, 0)));

    // 0 is reserved for empty space
    objects.push_back(nullptr);
  }

  virtual ~Grid() = default;

  inline char add_object(GridObject* obj) {
    if (obj->location.r >= height || obj->location.c >= width || obj->location.layer >= GridLayer::GridLayerCount) {
      return false;
    }
    if (this->grid[obj->location.r][obj->location.c][obj->location.layer] != 0) {
      return false;
    }

    obj->id = this->objects.size();
    this->objects.push_back(std::unique_ptr<GridObject>(obj));
    this->grid[obj->location.r][obj->location.c][obj->location.layer] = obj->id;
    return true;
  }

  // Removes and object from the grid and gives ownership of the object to the caller.
  // Since the caller is now the owner, this can make the raw pointer invalid, if the
  // returned unique_ptr is destroyed.
  inline unique_ptr<GridObject> remove_object(GridObject* obj) {
    this->grid[obj->location.r][obj->location.c][obj->location.layer] = 0;
    auto obj_ptr = this->objects[obj->id].release();
    this->objects[obj->id] = nullptr;
    return std::unique_ptr<GridObject>(obj_ptr);
  }

  inline char move_object(GridObjectId id, const GridLocation& loc) {
    if (loc.r >= height || loc.c >= width || loc.layer >= GridLayer::GridLayerCount) {
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
    GridLocation loc1 = obj1->location;
    Layer layer1 = loc1.layer;
    grid[loc1.r][loc1.c][loc1.layer] = 0;

    GridObject* obj2 = object(id2);
    GridLocation loc2 = obj2->location;
    Layer layer2 = loc2.layer;
    grid[loc2.r][loc2.c][loc2.layer] = 0;

    // Keep the layer the same
    obj1->location = loc2;
    obj1->location.layer = layer1;
    obj2->location = loc1;
    obj2->location.layer = layer2;

    grid[obj1->location.r][obj1->location.c][obj1->location.layer] = id1;
    grid[obj2->location.r][obj2->location.c][obj2->location.layer] = id2;
  }

  inline GridObject* object(GridObjectId obj_id) {
    return objects[obj_id].get();
  }

  inline GridObject* object_at(const GridLocation& loc) {
    if (loc.r >= height || loc.c >= width || loc.layer >= GridLayer::GridLayerCount) {
      return nullptr;
    }
    if (grid[loc.r][loc.c][loc.layer] == 0) {
      return nullptr;
    }
    return object(grid[loc.r][loc.c][loc.layer]);
  }

  inline const GridLocation location(GridObjectId id) {
    return object(id)->location;
  }

  inline const GridLocation relative_location(const GridLocation& loc,
                                              Orientation facing,
                                              short forward_distance,  // + is forward, - is backward
                                              short lateral_offset) {  // + is relative right, - is relative left
    const int r = static_cast<int>(loc.r);
    const int c = static_cast<int>(loc.c);

    int new_r;
    int new_c;

    switch (facing) {
      case Up:
        new_r = r - forward_distance;  // Positive dist = go up (decrease row)
        new_c = c + lateral_offset;    // Positive offset = go right (increase col)
        break;
      case Down:
        new_r = r + forward_distance;  // Positive dist = go down (increase row)
        new_c = c - lateral_offset;    // Positive offset = go left (decrease col)
        break;
      case Left:
        new_c = c - forward_distance;  // Positive dist = go left (decrease col)
        new_r = r - lateral_offset;    // Positive offset = go up (decrease row)
        break;
      case Right:
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

  inline const GridLocation relative_location(const GridLocation& loc, Orientation orientation) {
    return this->relative_location(loc, orientation, 1, 0);
  }

  inline char is_empty(unsigned int row, unsigned int col) {
    GridLocation loc;
    loc.r = row;
    loc.c = col;
    for (int layer = 0; layer < GridLayer::GridLayerCount; ++layer) {
      loc.layer = layer;
      if (object_at(loc) != nullptr) {
        return 0;
      }
    }
    return 1;
  }
};

#endif  // GRID_HPP_
