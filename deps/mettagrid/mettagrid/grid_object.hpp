#ifndef GRID_OBJECT_HPP
#define GRID_OBJECT_HPP

#include <string>
#include <vector>

using namespace std;

typedef unsigned short Layer;
typedef unsigned short TypeId;
typedef unsigned int GridCoord;
typedef unsigned char ObsType;

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

typedef unsigned int GridObjectId;

class GridObject {
public:
  GridObjectId id;
  GridLocation location;
  TypeId _type_id;

  virtual ~GridObject() = default;

  void init(TypeId type_id, const GridLocation& loc) {
    this->_type_id = type_id;
    this->location = loc;
  }

  void init(TypeId type_id, GridCoord r, GridCoord c) {
    init(type_id, GridLocation(r, c, 0));
  }

  void init(TypeId type_id, GridCoord r, GridCoord c, Layer layer) {
    init(type_id, GridLocation(r, c, layer));
  }

  virtual void obs(ObsType* obs, const vector<unsigned int>& offsets) const = 0;
};

#endif  // GRID_OBJECT_HPP
