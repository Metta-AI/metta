#ifndef GRID_OBJECT_HPP_
#define GRID_OBJECT_HPP_

#include <cstdint>
#include <span>
#include <string>
#include <vector>

#include "objects/constants.hpp"
#include "types.hpp"

using Layer = ObservationType;
using TypeId = ObservationType;
using ObservationCoord = ObservationType;

struct PartialObservationToken {
  ObservationType feature_id = EmptyTokenByte;
  ObservationType value = EmptyTokenByte;
};

// These may make more sense in observation_encoder.hpp, but we need to include that
// header in a lot of places, and it's nice to have these types defined in one place.
struct alignas(1) ObservationToken {
  ObservationType location = EmptyTokenByte;
  ObservationType feature_id = EmptyTokenByte;
  ObservationType value = EmptyTokenByte;
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

struct GridObjectConfig {
  TypeId type_id;
  std::string type_name;

  GridObjectConfig(TypeId type_id, const std::string& type_name) : type_id(type_id), type_name(type_name) {}

  virtual ~GridObjectConfig() = default;
};

class GridObject {
public:
  GridObjectId id{};
  GridLocation location{};
  TypeId type_id{};
  std::string type_name;

  GridObject() : id(0), location(), type_id(0), type_name() {}

  virtual ~GridObject() = default;

  void init(TypeId object_type_id, const std::string& object_type_name, const GridLocation& object_location) {
    this->type_id = object_type_id;
    this->type_name = object_type_name;
    this->location = object_location;
  }

  virtual bool swappable() const {
    return false;
  }

  virtual std::vector<PartialObservationToken> obs_features() const {
    return {};  // Default: no observable features
  }
};

#endif  // GRID_OBJECT_HPP_
