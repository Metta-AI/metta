// build_config.hpp
#ifndef PACKAGES_METTAGRID_CPP_INCLUDE_METTAGRID_ACTIONS_BUILD_CONFIG_HPP_
#define PACKAGES_METTAGRID_CPP_INCLUDE_METTAGRID_ACTIONS_BUILD_CONFIG_HPP_

#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include <string>
#include <unordered_map>
#include <vector>

#include "actions/action_handler.hpp"
#include "core/types.hpp"

// Holds the cost and object type for a vibe-triggered build
struct VibeBuildEffect {
  // Resource costs to pay (e.g., {energy: 10, carbon: 5})
  std::unordered_map<InventoryItem, InventoryQuantity> cost;
  // Key in objects map to construct
  std::string object_key;

  VibeBuildEffect() = default;
  VibeBuildEffect(const std::unordered_map<InventoryItem, InventoryQuantity>& cost, const std::string& object_key)
      : cost(cost), object_key(object_key) {}
};

struct BuildActionConfig : public ActionConfig {
  // Maps vibe ID to build effects (cost + object to construct)
  std::unordered_map<ObservationType, VibeBuildEffect> vibe_builds;
  bool enabled;
  std::vector<ObservationType> vibes;  // Vibes that trigger this action on move

  BuildActionConfig(const std::unordered_map<InventoryItem, InventoryQuantity>& required_resources = {},
                    const std::unordered_map<InventoryItem, InventoryQuantity>& consumed_resources = {},
                    const std::unordered_map<ObservationType, VibeBuildEffect>& vibe_builds = {},
                    bool enabled = true,
                    const std::vector<ObservationType>& vibes = {})
      : ActionConfig(required_resources, consumed_resources),
        vibe_builds(vibe_builds),
        enabled(enabled),
        vibes(vibes) {}
};

namespace py = pybind11;

inline void bind_vibe_build_effect(py::module& m) {
  py::class_<VibeBuildEffect>(m, "VibeBuildEffect")
      .def(py::init<>())
      .def(py::init<const std::unordered_map<InventoryItem, InventoryQuantity>&, const std::string&>(),
           py::arg("cost") = std::unordered_map<InventoryItem, InventoryQuantity>(),
           py::arg("object_key") = std::string())
      .def_readwrite("cost", &VibeBuildEffect::cost)
      .def_readwrite("object_key", &VibeBuildEffect::object_key);
}

inline void bind_build_action_config(py::module& m) {
  py::class_<BuildActionConfig, ActionConfig, std::shared_ptr<BuildActionConfig>>(m, "BuildActionConfig")
      .def(py::init<const std::unordered_map<InventoryItem, InventoryQuantity>&,
                    const std::unordered_map<InventoryItem, InventoryQuantity>&,
                    const std::unordered_map<ObservationType, VibeBuildEffect>&,
                    bool,
                    const std::vector<ObservationType>&>(),
           py::arg("required_resources") = std::unordered_map<InventoryItem, InventoryQuantity>(),
           py::arg("consumed_resources") = std::unordered_map<InventoryItem, InventoryQuantity>(),
           py::arg("vibe_builds") = std::unordered_map<ObservationType, VibeBuildEffect>(),
           py::arg("enabled") = true,
           py::arg("vibes") = std::vector<ObservationType>())
      .def_readwrite("vibe_builds", &BuildActionConfig::vibe_builds)
      .def_readwrite("enabled", &BuildActionConfig::enabled)
      .def_readwrite("vibes", &BuildActionConfig::vibes);
}

#endif  // PACKAGES_METTAGRID_CPP_INCLUDE_METTAGRID_ACTIONS_BUILD_CONFIG_HPP_
