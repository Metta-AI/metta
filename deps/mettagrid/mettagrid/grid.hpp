#ifndef GRID_HPP
#define GRID_HPP

#include <algorithm>
#include <vector>

#include "grid_object.hpp"

using namespace std;
typedef vector<vector<vector<GridObjectId> > > GridType;

class Grid {
public:
  unsigned int width;
  unsigned int height;
  vector<Layer> layer_for_type_id;
  Layer num_layers;

  GridType grid;
  vector<GridObject*> objects;

  inline Grid(unsigned int width, unsigned int height, vector<Layer> layer_for_type_id)
      : width(width), height(height), layer_for_type_id(layer_for_type_id) {
    num_layers = *max_element(layer_for_type_id.begin(), layer_for_type_id.end()) + 1;
    grid.resize(height, vector<vector<GridObjectId> >(width, vector<GridObjectId>(this->num_layers, 0)));

    // 0 is reserved for empty space
    objects.push_back(nullptr);
  }

  inline ~Grid() {
    for (unsigned long id = 1; id < objects.size(); ++id) {
      if (objects[id] != nullptr) {
        delete objects[id];
      }
    }
  }

  inline char add_object(GridObject* obj) {
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
    // delete obj;
    this->objects[obj->id] = nullptr;
  }

  inline void remove_object(GridObjectId id) {
    GridObject* obj = this->objects[id];
    this->remove_object(obj);
  }

  inline char move_object(GridObjectId id, const GridLocation& loc) {
    if (loc.r >= height or loc.c >= width or loc.layer >= num_layers) {
      return false;
    }

    if (grid[loc.r][loc.c][loc.layer] != 0) {
      return false;
    }

    GridObject* obj = objects[id];
    grid[loc.r][loc.c][loc.layer] = id;
    grid[obj->location.r][obj->location.c][obj->location.layer] = 0;
    obj->location = loc;
    return true;
  }

  inline void swap_objects(GridObjectId id1, GridObjectId id2) {
    GridObject* obj1 = objects[id1];
    GridLocation loc1 = obj1->location;
    Layer layer1 = loc1.layer;
    grid[loc1.r][loc1.c][loc1.layer] = 0;

    GridObject* obj2 = objects[id2];
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
    return objects[obj_id];
  }

  inline GridObject* object_at(const GridLocation& loc) {
    if (loc.r >= height or loc.c >= width or loc.layer >= num_layers) {
      return nullptr;
    }
    if (grid[loc.r][loc.c][loc.layer] == 0) {
      return nullptr;
    }
    return objects[grid[loc.r][loc.c][loc.layer]];
  }

  inline GridObject* object_at(const GridLocation& loc, TypeId type_id) {
    GridObject* obj = object_at(loc);
    if (obj != NULL and obj->_type_id == type_id) {
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
    return GridLocation(new_r, new_c, loc.layer);
  }

  inline const GridLocation relative_location(const GridLocation& loc,
                                              Orientation orientation,
                                              GridCoord distance,
                                              GridCoord offset,
                                              TypeId type_id) {
    GridLocation rloc = this->relative_location(loc, orientation, distance, offset);
    rloc.layer = this->layer_for_type_id[type_id];
    return rloc;
  }

  inline const GridLocation relative_location(const GridLocation& loc, Orientation orientation) {
    return this->relative_location(loc, orientation, 1, 0);
  }

  inline const GridLocation relative_location(const GridLocation& loc, Orientation orientation, TypeId type_id) {
    return this->relative_location(loc, orientation, 1, 0, type_id);
  }

  inline char is_empty(unsigned int row, unsigned int col) {
    GridLocation loc;
    loc.r = row;
    loc.c = col;
    for (int layer = 0; layer < num_layers; ++layer) {
      loc.layer = layer;
      if (object_at(loc) != nullptr) {
        return 0;
      }
    }
    return 1;
  }
};

#endif  // GRID_HPP
