#ifndef GRID_OBJECT_HPP
#define GRID_OBJECT_HPP

#include <cstdint>
#include <string>
#include <vector>

// Use the same typedefs as in core.hpp
typedef uint32_t GridObjectId;
typedef uint8_t ObsType;

// Layer and TypeId should be consistent
typedef uint16_t Layer;
typedef uint16_t TypeId;

// GridCoord should be consistent with the grid dimensions
typedef uint32_t GridCoord;

class GridLocation {
public:
  GridCoord r;
  GridCoord c;
  Layer layer;

  inline GridLocation(GridCoord r, GridCoord c, Layer layer) : r(r), c(c), layer(layer) {}
  inline GridLocation(GridCoord r, GridCoord c) : r(r), c(c), layer(0) {}
  inline GridLocation() : r(0), c(0), layer(0) {}
};

enum Orientation {
  Up = 0,
  Down = 1,
  Left = 2,
  Right = 3
};

class GridObject {
public:
  GridObjectId id;
  GridLocation location;
  TypeId _type_id;  // Changed to _type_id to match usage in grid.hpp

  virtual ~GridObject() = default;

  void init(TypeId type_id, const GridLocation& loc) {
    this->_type_id = type_id;  // Updated to use _type_id
    this->location = loc;
  }

  void init(TypeId type_id, GridCoord r, GridCoord c) {
    init(type_id, GridLocation(r, c, 0));
  }

  void init(TypeId type_id, GridCoord r, GridCoord c, Layer layer) {
    init(type_id, GridLocation(r, c, layer));
  }

  virtual void obs(ObsType* obs, const std::vector<uint32_t>& offsets) const = 0;  // Updated to use std:: and uint32_t
};

#endif  // GRID_OBJECT_HPP