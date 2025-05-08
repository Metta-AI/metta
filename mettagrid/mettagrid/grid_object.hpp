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
  static std::unordered_map<std::string, int> _feature_map;
  static int _next_feature_index;

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
    return _feature_map.size();
  }

  // Get all registered feature names
  static std::vector<std::string> get_feature_names() {
    std::vector<std::string> names(_feature_map.size());
    for (const auto& pair : _feature_map) {
      names[pair.second] = pair.first;
    }
    return names;
  }

protected:
  // Register a feature name if it's not already registered
  static size_t register_feature(const std::string& feature_name) {
    auto it = _feature_map.find(feature_name);
    if (it == _feature_map.end()) {
      // Feature not found, register it
      size_t index = _next_feature_index++;
      _feature_map[feature_name] = index;
      return index;
    }
    return it->second;
  }

  template <typename T>
  void encode(ObsType* obs, const std::string& feature_name, T value) const {
    // Register the feature if it doesn't exist
    size_t index = register_feature(feature_name);

    // Set the value in the observation array
    obs[index] = static_cast<ObsType>(value);
  }

  // Special handling for boolean values
  void encode(ObsType* obs, const std::string& feature_name, bool value) const {
    encode(obs, feature_name, value ? 1 : 0);
  }
};

#endif  // GRID_OBJECT_HPP