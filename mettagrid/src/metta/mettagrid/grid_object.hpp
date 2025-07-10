#ifndef GRID_OBJECT_HPP_
#define GRID_OBJECT_HPP_

#include <cstdint>
#include <span>
#include <string>
#include <vector>

#include "types.hpp"

using Layer = ObservationType;
using TypeId = ObservationType;
using GridCoord = ObservationType;

struct PartialObservationToken {
  ObservationType feature_id;
  ObservationType value;
};

// These may make more sense in observation_encoder.hpp, but we need to include that
// header in a lot of places, and it's nice to have these types defined in one place.
struct alignas(1) ObservationToken {
  ObservationType location;
  ObservationType feature_id;
  ObservationType value;
};

// The alignas should make sure of this, but let's be explicit.
// We're going to be reinterpret_casting things to this type, so
// it'll be bad if the compiler pads this type.
static_assert(sizeof(ObservationToken) == 3 * sizeof(uint8_t), "ObservationToken must be 3 bytes");

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

using GridObjectId = unsigned int;

struct GridObjectConfig {
  TypeId type_id;
  std::string type_name;

  GridObjectConfig(TypeId type_id, const std::string& type_name) : type_id(type_id), type_name(type_name) {}

  virtual ~GridObjectConfig() = default;
};

class GridObject {
public:
  GridObjectId id;
  GridLocation location;
  TypeId type_id;
  std::string type_name;

  virtual ~GridObject() = default;

  void init(TypeId type_id, const std::string& type_name, const GridLocation& loc) {
    this->type_id = type_id;
    this->type_name = type_name;
    this->location = loc;
  }

  virtual std::vector<PartialObservationToken> obs_features() const = 0;
};

#endif  // GRID_OBJECT_HPP_
