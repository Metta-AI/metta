#ifndef GRID_OBJECT_HPP_
#define GRID_OBJECT_HPP_

#include <cstdint>
#include <span>
#include <string>
#include <vector>

#include "packed_coordinate.hpp"
#include "types.hpp"

typedef unsigned short Layer;
typedef uint8_t TypeId;
typedef unsigned int GridCoord;

using InventoryItem = uint8_t;

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
static_assert(sizeof(ObservationToken) == 3, "ObservationToken must be 3 bytes");

using ObservationTokens = std::span<ObservationToken>;

struct PartialObservationToken {
  ObservationType feature_id;
  ObservationType value;
};

class GridLocation {
public:
  GridCoord r;
  GridCoord c;
  Layer layer;

  inline GridLocation(GridCoord r, GridCoord c, Layer layer) : r(r), c(c), layer(layer) {}
  inline GridLocation(GridCoord r, GridCoord c) : r(r), c(c), layer(0) {}
  inline GridLocation() : r(0), c(0), layer(0) {}

  /**
   * Pack this location's row and column into a single byte.
   * Note: This discards the layer information.
   *
   * @return Packed coordinate byte
   * @throws std::invalid_argument if r or c > 15
   */
  inline uint8_t pack() const {
    return PackedCoordinate::pack(static_cast<uint8_t>(r), static_cast<uint8_t>(c));
  }

  /**
   * Create a GridLocation from a packed coordinate if not empty.
   *
   * @param packed Packed coordinate byte
   * @param layer Layer to use (default 0)
   * @return std::optional<GridLocation> or std::nullopt if packed is empty
   */
  inline static std::optional<GridLocation> from_packed(uint8_t packed, Layer layer = 0) {
    auto unpacked = PackedCoordinate::unpack(packed);
    if (unpacked.has_value()) {
      return GridLocation(unpacked->first, unpacked->second, layer);
    }
    return std::nullopt;
  }
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
