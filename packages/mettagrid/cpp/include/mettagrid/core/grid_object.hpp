#ifndef PACKAGES_METTAGRID_CPP_INCLUDE_METTAGRID_CORE_GRID_OBJECT_HPP_
#define PACKAGES_METTAGRID_CPP_INCLUDE_METTAGRID_CORE_GRID_OBJECT_HPP_

#include <cstdint>
#include <span>
#include <string>
#include <vector>

#include "core/types.hpp"
#include "objects/constants.hpp"

using Layer = ObservationType;
using TypeId = ObservationType;
using ObservationCoord = ObservationType;

struct PartialObservationToken {
  ObservationType feature_id = EmptyTokenByte;
  ObservationType value = EmptyTokenByte;
};

struct alignas(1) ObservationToken {
  ObservationType location = EmptyTokenByte;
  ObservationType feature_id = EmptyTokenByte;
  ObservationType value = EmptyTokenByte;
};

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

  inline bool operator==(const GridLocation& other) const {
    return r == other.r && c == other.c && layer == other.layer;
  }
};

struct GridObjectConfig {
  TypeId type_id;
  std::string type_name;
  std::vector<int> tag_ids;

  GridObjectConfig(TypeId type_id, const std::string& type_name, const std::vector<int>& tag_ids = {})
      : type_id(type_id), type_name(type_name), tag_ids(tag_ids) {}

  virtual ~GridObjectConfig() = default;
};

class GridObject {
public:
  GridObjectId id{};
  TypeId type_id{};
  std::string type_name;
  std::vector<int> tag_ids;
  // All occupied locations. May be empty (off-grid).
  std::vector<GridLocation> locations;
  // Previous locations preserved across deactivation for reactivation
  std::vector<GridLocation> previous_locations;
  bool present_on_grid{false};

  virtual ~GridObject() = default;

  // Init without initial locations
  void init(TypeId object_type_id, const std::string& object_type_name, const std::vector<int>& tags) {
    this->type_id = object_type_id;
    this->type_name = object_type_name;
    this->tag_ids = tags;
    this->present_on_grid = false;
  }

  // Backward-compatible init that seeds with a single initial location
  void init(TypeId object_type_id,
            const std::string& object_type_name,
            const GridLocation& initial_location,
            const std::vector<int>& tags) {
    this->type_id = object_type_id;
    this->type_name = object_type_name;
    this->tag_ids = tags;
    this->present_on_grid = false;
    this->locations.clear();
    this->locations.push_back(initial_location);
    this->previous_locations = this->locations;
  }

  virtual bool swappable() const {
    return false;
  }
  virtual std::vector<PartialObservationToken> obs_features() const {
    return {};
  }

  // Whether this object type supports occupying multiple cells.
  // Default: false. Only Assembler overrides to true.
  virtual bool supports_multi_cell() const {
    return false;
  }
};

#endif  // PACKAGES_METTAGRID_CPP_INCLUDE_METTAGRID_CORE_GRID_OBJECT_HPP_
