#ifndef PACKAGES_METTAGRID_CPP_INCLUDE_METTAGRID_OBJECTS_WALL_HPP_
#define PACKAGES_METTAGRID_CPP_INCLUDE_METTAGRID_OBJECTS_WALL_HPP_

#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include <string>
#include <vector>

#include "config/observation_features.hpp"
#include "core/grid_object.hpp"
#include "objects/constants.hpp"

// #MettaGridConfig
struct WallConfig : public GridObjectConfig {
  WallConfig(TypeId type_id, const std::string& type_name, ObservationType initial_vibe = 0)
      : GridObjectConfig(type_id, type_name, initial_vibe) {}
};

class Wall : public GridObject {
public:
  Wall(GridCoord r, GridCoord c, const WallConfig& cfg) {
    GridObject::init(cfg.type_id, cfg.type_name, GridLocation(r, c), cfg.tag_ids, cfg.initial_vibe);
  }

  std::vector<PartialObservationToken> obs_features() const override {
    std::vector<PartialObservationToken> features;
    features.reserve(1 + tag_ids.size() + (this->vibe != 0 ? 1 : 0));

    // Emit tag features
    for (int tag_id : tag_ids) {
      features.push_back({ObservationFeature::Tag, static_cast<ObservationType>(tag_id)});
    }

    if (this->vibe != 0) features.push_back({ObservationFeature::Vibe, static_cast<ObservationType>(this->vibe)});

    return features;
  }
};

namespace py = pybind11;

inline void bind_wall_config(py::module& m) {
  py::class_<WallConfig, GridObjectConfig, std::shared_ptr<WallConfig>>(m, "WallConfig")
      .def(py::init<TypeId, const std::string&, ObservationType>(),
           py::arg("type_id"),
           py::arg("type_name"),
           py::arg("initial_vibe") = 0)
      .def_readwrite("type_id", &WallConfig::type_id)
      .def_readwrite("type_name", &WallConfig::type_name)
      .def_readwrite("tag_ids", &WallConfig::tag_ids)
      .def_readwrite("initial_vibe", &WallConfig::initial_vibe);
}

#endif  // PACKAGES_METTAGRID_CPP_INCLUDE_METTAGRID_OBJECTS_WALL_HPP_
