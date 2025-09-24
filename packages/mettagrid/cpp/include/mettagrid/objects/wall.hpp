#ifndef PACKAGES_METTAGRID_CPP_INCLUDE_METTAGRID_OBJECTS_WALL_HPP_
#define PACKAGES_METTAGRID_CPP_INCLUDE_METTAGRID_OBJECTS_WALL_HPP_

#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include <string>
#include <vector>

#include "core/grid_object.hpp"
#include "objects/constants.hpp"

// #MettaGridConfig
struct WallConfig : public GridObjectConfig {
  WallConfig(TypeId type_id, const std::string& type_name, bool swappable, const std::vector<int>& tag_ids = {})
      : GridObjectConfig(type_id, type_name, tag_ids), swappable(swappable) {}

  bool swappable;
};

class Wall : public GridObject {
public:
  bool _swappable;

  Wall(GridCoord r, GridCoord c, const WallConfig& cfg) {
    GridObject::init(cfg.type_id, cfg.type_name, GridLocation(r, c, GridLayer::ObjectLayer), cfg.tag_ids);
    this->_swappable = cfg.swappable;
  }

  std::vector<PartialObservationToken> obs_features() const override {
    std::vector<PartialObservationToken> features;
    features.reserve(2 + tag_ids.size());
    features.push_back({ObservationFeature::TypeId, static_cast<ObservationType>(this->type_id)});

    if (_swappable) {
      // Only emit the swappable observation feature when True to reduce the number of tokens.
      features.push_back({ObservationFeature::Swappable, static_cast<ObservationType>(1)});
    }

    // Emit tag features
    for (int tag_id : tag_ids) {
      features.push_back({ObservationFeature::Tag, static_cast<ObservationType>(tag_id)});
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
      .def(py::init<TypeId, const std::string&, bool, const std::vector<int>&>(),
           py::arg("type_id"), py::arg("type_name"), py::arg("swappable"), py::arg("tag_ids") = std::vector<int>{})
      .def_readwrite("type_id", &WallConfig::type_id)
      .def_readwrite("type_name", &WallConfig::type_name)
      .def_readwrite("swappable", &WallConfig::swappable)
      .def_readwrite("tag_ids", &WallConfig::tag_ids);
}

#endif  // PACKAGES_METTAGRID_CPP_INCLUDE_METTAGRID_OBJECTS_WALL_HPP_
