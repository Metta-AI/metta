#ifndef GRID_OBJECT_HPP
#define GRID_OBJECT_HPP

#include <cstdint>
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

  // obs_tokens is used for the new observation format.
  // `prefix` will prefix each token, but in practice should indicate the observation location.
  // `feature_ids` should map to the features that the object is encoding, and should match the
  // features exposed by feature_names.
  // `max_tokens` is the maximum number of tokens that can be encoded. This should be used
  // to stop us from overflowing the observation buffer. Note that each token is multiple bytes,
  // so max_tokens should not just be the free space in the observation buffer.
  virtual void obs_tokens(ObsType* obs, ObsType prefix, const vector<unsigned char>& feature_ids, size_t max_tokens) const = 0;

  virtual void obs(ObsType* obs, const vector<uint8_t>& offsets) const = 0;
};

#endif  // GRID_OBJECT_HPP
