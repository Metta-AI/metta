#ifndef GRID_HPP
#define GRID_HPP

#include <algorithm>
#include <cstdint>
#include <vector>

#include "grid_object.hpp"

typedef std::vector<std::vector<std::vector<GridObjectId>>> GridType;

class Grid {
public:
  uint32_t width;
  uint32_t height;
  std::vector<Layer> layer_for_type_id;
  Layer num_layers;

  GridType grid;
  std::vector<GridObject*> objects;

  inline Grid(uint32_t width, uint32_t height, std::vector<Layer> layer_for_type_id)
      : width(width), height(height), layer_for_type_id(layer_for_type_id) {
    num_layers = *std::max_element(layer_for_type_id.begin(), layer_for_type_id.end()) + 1;
    grid.resize(height, std::vector<std::vector<GridObjectId>>(width, std::vector<GridObjectId>(this->num_layers, 0)));

    // 0 is reserved for empty space
    objects.push_back(nullptr);
  }

  inline ~Grid() {
    for (uint64_t id = 1; id < objects.size(); ++id) {
      if (objects[id] != nullptr) {
        delete objects[id];
        objects[id] = nullptr;  // Set to nullptr after deletion for safety
      }
    }
  }

  inline bool add_object(GridObject* obj) {
    if (obj->location.r >= height or obj->location.c >= width or obj->location.layer >= num_layers) {
      return false;
    }
    if (this->grid[obj->location.r][obj->location.c][obj->location.layer] != 0) {
      return false;
    }

    obj->id = this->objects.size();
    this->objects.push_back(obj);
    this->grid[obj->location.r][obj->location.c][obj->location.layer] = obj->id;
    return true;
  }

  inline void remove_object(GridObject* obj) {
    this->grid[obj->location.r][obj->location.c][obj->location.layer] = 0;
    // Don't delete the object here, just mark it as removed in the grid
    this->objects[obj->id] = nullptr;
    // The actual deletion is deferred to the destructor
  }

  inline void remove_object(GridObjectId id) {
    GridObject* obj = this->objects[id];
    this->remove_object(obj);
  }

  inline bool move_object(GridObjectId id, const GridLocation& dest) {
    if (dest.r >= height or dest.c >= width or dest.layer >= num_layers) {
      return false;
    }

    if (grid[dest.r][dest.c][dest.layer] != 0) {
      return false;
    }

    GridObject* obj = objects[id];
    GridLocation src = obj->location;
    grid[dest.r][dest.c][dest.layer] = id;
    grid[src.r][src.c][src.layer] = 0;
    obj->location = dest;
    return true;
  }

  inline void swap_objects(GridObjectId id1, GridObjectId id2) {
    GridObject* obj1 = objects[id1];
    GridLocation pos1 = obj1->location;
    Layer layer1 = pos1.layer;
    grid[pos1.r][pos1.c][pos1.layer] = 0;

    GridObject* obj2 = objects[id2];
    GridLocation pos2 = obj2->location;
    Layer layer2 = pos2.layer;
    grid[pos2.r][pos2.c][pos2.layer] = 0;

    // Keep the layer the same
    obj1->location = pos2;
    obj1->location.layer = layer1;
    obj2->location = pos1;
    obj2->location.layer = layer2;

    grid[obj1->location.r][obj1->location.c][obj1->location.layer] = id1;
    grid[obj2->location.r][obj2->location.c][obj2->location.layer] = id2;
  }

  inline GridObject* object(GridObjectId obj_id) {
    return objects[obj_id];
  }

  inline GridObject* object_at(const GridLocation& pos) {
    if (pos.r >= height or pos.c >= width or pos.layer >= num_layers) {
      return nullptr;
    }
    if (grid[pos.r][pos.c][pos.layer] == 0) {
      return nullptr;
    }
    return objects[grid[pos.r][pos.c][pos.layer]];
  }

  inline GridObject* object_at(const GridLocation& pos, TypeId type_id) {
    GridObject* obj = object_at(pos);
    if (obj != nullptr and obj->_type_id == type_id) {
      return obj;
    }
    return nullptr;
  }

  inline GridObject* object_at(GridCoord r, GridCoord c, TypeId type_id) {
    GridObject* obj = object_at(GridLocation(r, c), this->layer_for_type_id[type_id]);
    if (obj->_type_id != type_id) {
      return nullptr;
    }

    return obj;
  }

  inline const GridLocation location(GridObjectId id) {
    return objects[id]->location;
  }

  inline const GridLocation relative_location(const GridLocation& origin,
                                              Orientation orientation,
                                              GridCoord distance,
                                              GridCoord offset) {
    int32_t targetR = origin.r;
    int32_t targetC = origin.c;

    switch (orientation) {
      case Up:
        targetR = origin.r - distance;
        targetC = origin.c - offset;
        break;
      case Down:
        targetR = origin.r + distance;
        targetC = origin.c + offset;
        break;
      case Left:
        targetR = origin.r + offset;
        targetC = origin.c - distance;
        break;
      case Right:
        targetR = origin.r - offset;
        targetC = origin.c + distance;
        break;
    }
    targetR = std::max(0, targetR);
    targetC = std::max(0, targetC);
    return GridLocation(targetR, targetC, origin.layer);
  }

  inline const GridLocation relative_location(const GridLocation& origin,
                                              Orientation orientation,
                                              GridCoord distance,
                                              GridCoord offset,
                                              TypeId type_id) {
    GridLocation target = this->relative_location(origin, orientation, distance, offset);
    target.layer = this->layer_for_type_id[type_id];
    return target;
  }

  inline const GridLocation relative_location(const GridLocation& origin, Orientation orientation) {
    return this->relative_location(origin, orientation, 1, 0);
  }

  inline const GridLocation relative_location(const GridLocation& origin, Orientation orientation, TypeId type_id) {
    return this->relative_location(origin, orientation, 1, 0, type_id);
  }

  inline bool is_empty(uint32_t row, uint32_t col) {
    GridLocation pos;
    pos.r = row;
    pos.c = col;
    for (int32_t layer = 0; layer < num_layers; ++layer) {
      pos.layer = layer;
      if (object_at(pos) != nullptr) {
        return false;
      }
    }
    return true;
  }
};

#endif  // GRID_HPP