#ifndef GRID_OBJECT_HPP
#define GRID_OBJECT_HPP

#include <cstdint>
#include <string>
#include <vector>
#include <span>
using namespace std;

typedef unsigned short Layer;
typedef unsigned short TypeId;
typedef unsigned int GridCoord;
using ObsType = uint8_t;

// These may make more sense in observation_encoder.hpp, but we need to include that
// header in a lot of places, and it's nice to have these types defined in one place.
struct alignas(1) ObservationToken {
  uint8_t prefix;
  uint8_t feature_id;
  uint8_t value;
};

// The alignas should make sure of this, but let's be explicit.
// We're going to be reinterpret_casting things to this type, so
// it'll be bad if the compiler pads this type.
static_assert(sizeof(ObservationToken) == 3, "ObservationToken must be 3 bytes");

using ObservationTokens = std::span<ObservationToken>;

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
  // `feature_ids` should map to the features that the object is encoding, and should match the
  // features exposed by feature_names.
  virtual size_t obs_tokens(ObservationTokens tokens,
                             const vector<unsigned char>& feature_ids) const = 0;

  virtual void obs(ObsType* obs, const vector<uint8_t>& offsets) const = 0;
};

#endif  // GRID_OBJECT_HPP
