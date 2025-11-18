#ifndef PACKAGES_METTAGRID_CPP_INCLUDE_METTAGRID_CORE_GRID_OBJECT_HPP_
#define PACKAGES_METTAGRID_CPP_INCLUDE_METTAGRID_CORE_GRID_OBJECT_HPP_

#include <cassert>
#include <cstdint>
#include <span>
#include <string>
#include <utility>
#include <vector>

#include "core/types.hpp"
#include "objects/constants.hpp"
#include "objects/has_vibe.hpp"

using TypeId = ObservationType;
using ObservationCoord = ObservationType;
using Vibe = ObservationType;

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

  inline GridLocation(GridCoord r, GridCoord c) : r(r), c(c) {}
  inline GridLocation() : r(0), c(0) {}

  inline bool operator==(const GridLocation& other) const {
    return r == other.r && c == other.c;
  }
};

struct GridObjectConfig {
  TypeId type_id;
  std::string type_name;
  std::vector<int> tag_ids;
  ObservationType initial_vibe;

  GridObjectConfig(TypeId type_id, const std::string& type_name, ObservationType initial_vibe = 0)
      : type_id(type_id), type_name(type_name), tag_ids({}), initial_vibe(initial_vibe) {}

  virtual ~GridObjectConfig() = default;
};

class GridObject : public HasVibe {
public:
  GridObjectId id{};
  TypeId type_id{};
  std::string type_name;
  std::vector<int> tag_ids;

  // All occupied locations for this object. For single-cell objects this
  // contains exactly one entry. For multi-cell objects the first entry is
  // the anchor cell and additional entries represent the rest of the
  // footprint. The footprint is seeded at construction time and remains
  // stable; movement for single-cell objects is expressed by updating
  // locations[0].
  std::vector<GridLocation> locations;

  virtual ~GridObject() = default;

protected:
  // Construct a GridObject with a fully-initialized, non-empty footprint.
  // All callers must provide at least one location; this guarantees that
  // helpers like location() and grid_objects() never see an empty footprint.
  GridObject(TypeId object_type_id,
             const std::string& object_type_name,
             std::vector<GridLocation> object_locations,
             const std::vector<int>& tags,
             ObservationType object_vibe = 0)
      : HasVibe(object_vibe),
        id(0),
        type_id(object_type_id),
        type_name(object_type_name),
        tag_ids(tags),
        locations(std::move(object_locations)) {
    assert(!locations.empty() && "GridObject constructed with empty locations");
  }

public:
  // Anchor location for single-cell semantics; requires the object to have
  // at least one populated location.
  const GridLocation& location() const {
    assert(!locations.empty() && "location() called on object with empty locations");
    return locations.front();
  }

  virtual std::vector<PartialObservationToken> obs_features() const {
    return {};  // Default: no observable features
  }

  virtual bool supports_multi_cell() const {
    return false;
  }
};

#endif  // PACKAGES_METTAGRID_CPP_INCLUDE_METTAGRID_CORE_GRID_OBJECT_HPP_
