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
    GridObject::init(cfg.type_id,
                     cfg.type_name,
                     GridLocation(r, c),
                     cfg.tag_ids,
                     cfg.initial_vibe,
                     cfg.aoes,
                     cfg.effective_name(),
                     cfg.activation_handlers);
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

inline void bind_activation_handler_config(py::module& m) {
  // Bind enums
  py::enum_<FilterType>(m, "FilterType")
      .value("VIBE", FilterType::VIBE)
      .value("RESOURCE", FilterType::RESOURCE)
      .value("ALIGNMENT", FilterType::ALIGNMENT)
      .value("TAG", FilterType::TAG);

  py::enum_<EntitySelector>(m, "EntitySelector")
      .value("ACTOR", EntitySelector::ACTOR)
      .value("TARGET", EntitySelector::TARGET)
      .value("ACTOR_FACTION", EntitySelector::ACTOR_FACTION)
      .value("TARGET_FACTION", EntitySelector::TARGET_FACTION);

  py::enum_<AlignmentCondition>(m, "AlignmentCondition")
      .value("ALIGNED", AlignmentCondition::ALIGNED)
      .value("UNALIGNED", AlignmentCondition::UNALIGNED)
      .value("SAME_FACTION", AlignmentCondition::SAME_FACTION)
      .value("DIFFERENT_FACTION", AlignmentCondition::DIFFERENT_FACTION);

  py::enum_<MutationType>(m, "MutationType")
      .value("RESOURCE_DELTA", MutationType::RESOURCE_DELTA)
      .value("RESOURCE_TRANSFER", MutationType::RESOURCE_TRANSFER)
      .value("ALIGNMENT", MutationType::ALIGNMENT)
      .value("FREEZE", MutationType::FREEZE)
      .value("ATTACK", MutationType::ATTACK);

  // Bind FilterConfig
  py::class_<FilterConfig, std::shared_ptr<FilterConfig>>(m, "FilterConfig")
      .def(py::init<>())
      .def_readwrite("type", &FilterConfig::type)
      .def_readwrite("entity", &FilterConfig::entity)
      .def_readwrite("vibe_value", &FilterConfig::vibe_value)
      .def_readwrite("resource_id", &FilterConfig::resource_id)
      .def_readwrite("min_amount", &FilterConfig::min_amount)
      .def_readwrite("alignment_condition", &FilterConfig::alignment_condition)
      .def_readwrite("tag_id", &FilterConfig::tag_id);

  // Bind MutationConfig
  py::class_<MutationConfig, std::shared_ptr<MutationConfig>>(m, "MutationConfig")
      .def(py::init<>())
      .def_readwrite("type", &MutationConfig::type)
      .def_readwrite("entity", &MutationConfig::entity)
      .def_readwrite("resource_deltas", &MutationConfig::resource_deltas)
      .def_readwrite("transfer_source", &MutationConfig::transfer_source)
      .def_readwrite("transfer_target", &MutationConfig::transfer_target)
      .def_readwrite("align_to_actor", &MutationConfig::align_to_actor)
      .def_readwrite("freeze_duration", &MutationConfig::freeze_duration)
      .def_readwrite("attack_damage", &MutationConfig::attack_damage);

  // Bind ActivationHandlerConfig
  py::class_<ActivationHandlerConfig, std::shared_ptr<ActivationHandlerConfig>>(m, "ActivationHandlerConfig")
      .def(py::init<>())
      .def(py::init<const std::string&, const std::vector<FilterConfig>&, const std::vector<MutationConfig>&>(),
           py::arg("name"),
           py::arg("filters"),
           py::arg("mutations"))
      .def_readwrite("name", &ActivationHandlerConfig::name)
      .def_readwrite("filters", &ActivationHandlerConfig::filters)
      .def_readwrite("mutations", &ActivationHandlerConfig::mutations);
}

inline void bind_wall_config(py::module& m) {
  py::class_<WallConfig, GridObjectConfig, std::shared_ptr<WallConfig>>(m, "WallConfig")
      .def(py::init<TypeId, const std::string&, ObservationType>(),
           py::arg("type_id"),
           py::arg("type_name"),
           py::arg("initial_vibe") = 0)
      .def_readwrite("type_id", &WallConfig::type_id)
      .def_readwrite("type_name", &WallConfig::type_name)
      .def_readwrite("name", &WallConfig::name)
      .def_readwrite("tag_ids", &WallConfig::tag_ids)
      .def_readwrite("initial_vibe", &WallConfig::initial_vibe)
      .def_readwrite("aoes", &WallConfig::aoes)
      .def_readwrite("activation_handlers", &WallConfig::activation_handlers);
}

#endif  // PACKAGES_METTAGRID_CPP_INCLUDE_METTAGRID_OBJECTS_WALL_HPP_
