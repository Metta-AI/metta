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
    GridObject::init(cfg.type_id, cfg.type_name, GridLocation(r, c), cfg.tag_ids, cfg.initial_vibe, cfg.aoes);
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

inline void bind_aoe_effect_config(py::module& m) {
  py::class_<AOEEffectConfig, std::shared_ptr<AOEEffectConfig>>(m, "AOEEffectConfig")
      .def(py::init<>())
      .def(py::init<unsigned int,
                    const std::unordered_map<InventoryItem, InventoryDelta>&,
                    const std::vector<int>&,
                    bool,
                    bool>(),
           py::arg("range") = 1,
           py::arg("resource_deltas") = std::unordered_map<InventoryItem, InventoryDelta>(),
           py::arg("target_tag_ids") = std::vector<int>(),
           py::arg("same_faction_only") = false,
           py::arg("different_faction_only") = false)
      .def_readwrite("range", &AOEEffectConfig::range)
      .def_readwrite("resource_deltas", &AOEEffectConfig::resource_deltas)
      .def_readwrite("target_tag_ids", &AOEEffectConfig::target_tag_ids)
      .def_readwrite("same_faction_only", &AOEEffectConfig::same_faction_only)
      .def_readwrite("different_faction_only", &AOEEffectConfig::different_faction_only);
}

inline void bind_wall_config(py::module& m) {
  py::class_<WallConfig, GridObjectConfig, std::shared_ptr<WallConfig>>(m, "WallConfig")
      .def(py::init<TypeId, const std::string&, ObservationType>(),
           py::arg("type_id"),
           py::arg("type_name"),
           py::arg("initial_vibe") = 0)
      .def_readwrite("type_id", &WallConfig::type_id)
      .def_readwrite("type_name", &WallConfig::type_name)
      .def_readwrite("tag_ids", &WallConfig::tag_ids)
      .def_readwrite("initial_vibe", &WallConfig::initial_vibe)
      .def_readwrite("aoes", &WallConfig::aoes);
}

#endif  // PACKAGES_METTAGRID_CPP_INCLUDE_METTAGRID_OBJECTS_WALL_HPP_
