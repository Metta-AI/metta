#ifndef OBJECTS_WALL_HPP_
#define OBJECTS_WALL_HPP_

#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include <string>
#include <vector>

#include "../grid_object.hpp"
#include "constants.hpp"

// #MettaGridConfig
struct WallConfig : public GridObjectConfig {
  WallConfig(TypeId type_id, const std::string& type_name, bool swappable)
      : GridObjectConfig(type_id, type_name), swappable(swappable) {}

  bool swappable;
};

class Wall : public GridObject {
public:
  bool _swappable;

  Wall(GridCoord r, GridCoord c, const WallConfig& cfg) {
    GridObject::init(cfg.type_id, cfg.type_name, GridLocation(r, c, GridLayer::ObjectLayer));
    this->_swappable = cfg.swappable;
  }

  std::vector<PartialObservationToken> obs_features() const override {
    std::vector<PartialObservationToken> features;
    features.reserve(2);
    features.push_back({ObservationFeature::TypeId, static_cast<ObservationType>(this->type_id)});

    if (_swappable) {
      // Only emit the swappable observation feature when True to reduce the number of tokens.
      features.push_back({ObservationFeature::Swappable, static_cast<ObservationType>(1)});
    }

    return features;
  }

  bool swappable() const override {
    return this->_swappable;
  }
};

namespace py = pybind11;

inline void bind_wall_config(py::module& m) {
  py::class_<WallConfig, GridObjectConfig, std::shared_ptr<WallConfig>>(m, "WallConfig")
      .def(py::init<TypeId, const std::string&, bool>(), py::arg("type_id"), py::arg("type_name"), py::arg("swappable"))
      .def_readwrite("type_id", &WallConfig::type_id)
      .def_readwrite("type_name", &WallConfig::type_name)
      .def_readwrite("swappable", &WallConfig::swappable);
}

#endif  // OBJECTS_WALL_HPP_
