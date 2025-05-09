#ifndef GRID_OBJECT_HPP
#define GRID_OBJECT_HPP

#include <cstdint>
#include <iostream>
#include <memory>
#include <string>
#include <unordered_map>
#include <vector>

#include "constants.hpp"
#include "types.hpp"

class GridLocation {
public:
  GridCoord r;
  GridCoord c;
  Layer layer;

  inline GridLocation(GridCoord r, GridCoord c, Layer layer) : r(r), c(c), layer(layer) {}
  inline GridLocation(GridCoord r, GridCoord c) : r(r), c(c), layer(0) {}
  inline GridLocation() : r(0), c(0), layer(0) {}
};

class GridObject {
private:
  inline static std::unordered_map<std::string, int> _feature_map{};
  inline static int _next_feature_index = 0;

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

  // Pure virtual method to be implemented by derived classes
  virtual void obs(ObsType* obs) const = 0;

  // Get the observation size (total number of features)
  static size_t get_observation_size() {
    return static_cast<size_t>(GridFeature::COUNT);
  }

  // Get all feature names
  static const std::vector<std::string>& get_feature_names() {
    return GridFeatureNames;
  }

protected:
  template <typename T>
  void encode(ObsType* obs, GridFeature feature, T value) const {
    // Set the value in the observation array at the specified feature index
    obs[static_cast<size_t>(feature)] = static_cast<ObsType>(value);
  }

  // Special handling for boolean values
  void encode(ObsType* obs, GridFeature feature, bool value) const {
    encode(obs, feature, value ? 1 : 0);
  }
};

#endif  // GRID_OBJECT_HPP