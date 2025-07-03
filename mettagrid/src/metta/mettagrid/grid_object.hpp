#ifndef GRID_OBJECT_HPP_
#define GRID_OBJECT_HPP_

#include <cstdint>
#include <span>
#include <string>
#include <vector>

// using namespace std;  // Removed per cpplint

typedef unsigned short Layer;
typedef uint8_t TypeId;
typedef unsigned int GridCoord;
using ObsType = uint8_t;
using InventoryItem = uint8_t;

// These may make more sense in observation_encoder.hpp, but we need to include that
// header in a lot of places, and it's nice to have these types defined in one place.
struct alignas(1) ObservationToken {
  uint8_t location;
  uint8_t feature_id;
  uint8_t value;
};

// The alignas should make sure of this, but let's be explicit.
// We're going to be reinterpret_casting things to this type, so
// it'll be bad if the compiler pads this type.
static_assert(sizeof(ObservationToken) == 3, "ObservationToken must be 3 bytes");

using ObservationTokens = std::span<ObservationToken>;

struct PartialObservationToken {
  uint8_t feature_id;
  uint8_t value;
};

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
  TypeId type_id;
  std::string type_name;

  virtual ~GridObject() = default;

  void init(TypeId object_type_id, const std::string& object_type_name, const GridLocation& object_location) {
    this->type_id = object_type_id;
    this->type_name = object_type_name;
    this->location = object_location;
  }

  virtual std::vector<PartialObservationToken> obs_features() const = 0;
};

#endif  // GRID_OBJECT_HPP_
