#ifndef GRID_OBJECT_HPP_
#define GRID_OBJECT_HPP_

#include <cstdint>
#include <span>
#include <string>
#include <vector>

#include "observation_tokens.hpp"
#include "types.hpp"

using Layer = ObservationType;
using TypeId = ObservationType;
using ObservationCoord = ObservationType;

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

  GridObjectConfig(TypeId type_id, const std::string& type_name) : type_id(type_id), type_name(type_name) {}

  virtual ~GridObjectConfig() = default;
};

class GridObject {
public:
  GridObjectId id{};
  GridLocation location{};
  TypeId type_id{};
  std::string type_name;

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
