#ifndef PACKAGES_METTAGRID_CPP_INCLUDE_METTAGRID_CONFIG_METTAGRID_CONFIG_HPP_
#define PACKAGES_METTAGRID_CPP_INCLUDE_METTAGRID_CONFIG_METTAGRID_CONFIG_HPP_

#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include <memory>
#include <string>
#include <unordered_map>
#include <vector>

#include "core/types.hpp"
#include "systems/clipper_config.hpp"

// Forward declarations
struct ActionConfig;
struct GridObjectConfig;

using ObservationCoord = ObservationType;

/**
 * Represents a global observation feature with optional parameters.
 * Supports both simple features (just a name) and parameterized features.
 */
struct GlobalObsFeature {
  std::string name;
  pybind11::dict params;  // Store as Python dict for simplicity

  // Constructor for simple features (name only)
  explicit GlobalObsFeature(const std::string& feature_name)
      : name(feature_name), params() {}

  // Constructor for parameterized features
  GlobalObsFeature(const std::string& feature_name, const pybind11::dict& feature_params)
      : name(feature_name), params(feature_params) {}

  // Check if feature is enabled (always true for items in the list)
  bool is_enabled() const { return true; }

  // Get integer parameter with default
  int get_int_param(const std::string& key, int default_value) const {
    if (params.contains(key)) {
      return params[key.c_str()].cast<int>();
    }
    return default_value;
  }

  // Get float parameter with default
  float get_float_param(const std::string& key, float default_value) const {
    if (params.contains(key)) {
      return params[key.c_str()].cast<float>();
    }
    return default_value;
  }

  // Get bool parameter with default
  bool get_bool_param(const std::string& key, bool default_value) const {
    if (params.contains(key)) {
      return params[key.c_str()].cast<bool>();
    }
    return default_value;
  }

  // Get string parameter with default
  std::string get_str_param(const std::string& key, const std::string& default_value) const {
    if (params.contains(key)) {
      return params[key.c_str()].cast<std::string>();
    }
    return default_value;
  }

  // Check if parameter exists
  bool has_param(const std::string& key) const {
    return params.contains(key);
  }
};

struct GlobalObsConfig {
  bool episode_completion_pct = true;
  bool last_action = true;
  bool last_reward = true;
  bool visitation_counts = false;
};

struct GameConfig {
  size_t num_agents;
  unsigned int max_steps;
  bool episode_truncates;
  ObservationCoord obs_width;
  ObservationCoord obs_height;
  std::vector<std::string> resource_names;
  unsigned int num_observation_tokens;

  // Global observations - NEW flexible system
  std::vector<GlobalObsFeature> global_observations;

  // Global observations - OLD struct (deprecated, for backwards compatibility)
  GlobalObsConfig global_obs;

  std::vector<std::pair<std::string, std::shared_ptr<ActionConfig>>> actions;  // Ordered list of (name, config) pairs
  std::unordered_map<std::string, std::shared_ptr<GridObjectConfig>> objects;
  float resource_loss_prob = 0.0;
  std::unordered_map<int, std::string> tag_id_map;

  // FEATURE FLAGS
  bool track_movement_metrics = false;
  bool recipe_details_obs = false;
  bool allow_diagonals = false;
  std::unordered_map<std::string, float> reward_estimates = {};

  // Inventory regeneration interval (global check timing)
  unsigned int inventory_regen_interval = 0;  // Interval in timesteps (0 = disabled)

  // Global clipper settings
  std::shared_ptr<ClipperConfig> clipper = nullptr;
};

namespace py = pybind11;

inline void bind_global_obs_feature(py::module& m) {
  py::class_<GlobalObsFeature>(m, "GlobalObsFeature")
      .def(py::init<const std::string&>(),
           py::arg("name"),
           "Create a simple global observation feature")
      .def(py::init<const std::string&, const pybind11::dict&>(),
           py::arg("name"),
           py::arg("params"),
           "Create a parameterized global observation feature")
      .def_readwrite("name", &GlobalObsFeature::name)
      .def_readwrite("params", &GlobalObsFeature::params)
      .def("is_enabled", &GlobalObsFeature::is_enabled)
      .def("has_param", &GlobalObsFeature::has_param, py::arg("key"))
      .def("get_int_param", &GlobalObsFeature::get_int_param, py::arg("key"), py::arg("default_value"))
      .def("get_float_param", &GlobalObsFeature::get_float_param, py::arg("key"), py::arg("default_value"))
      .def("get_bool_param", &GlobalObsFeature::get_bool_param, py::arg("key"), py::arg("default_value"))
      .def("get_str_param", &GlobalObsFeature::get_str_param, py::arg("key"), py::arg("default_value"));
}

inline void bind_global_obs_config(py::module& m) {
  py::class_<GlobalObsConfig>(m, "GlobalObsConfig")
      .def(py::init<>())
      .def(py::init<bool, bool, bool, bool>(),
           py::arg("episode_completion_pct") = true,
           py::arg("last_action") = true,
           py::arg("last_reward") = true,
           py::arg("visitation_counts") = false)
      .def_readwrite("episode_completion_pct", &GlobalObsConfig::episode_completion_pct)
      .def_readwrite("last_action", &GlobalObsConfig::last_action)
      .def_readwrite("last_reward", &GlobalObsConfig::last_reward)
      .def_readwrite("visitation_counts", &GlobalObsConfig::visitation_counts);
}

inline void bind_game_config(py::module& m) {
  py::class_<GameConfig>(m, "GameConfig")
      .def(py::init<unsigned int,
                    unsigned int,
                    bool,
                    ObservationCoord,
                    ObservationCoord,
                    const std::vector<std::string>&,
                    unsigned int,
                    const std::vector<GlobalObsFeature>&,
                    const GlobalObsConfig&,
                    const std::vector<std::pair<std::string, std::shared_ptr<ActionConfig>>>&,
                    const std::unordered_map<std::string, std::shared_ptr<GridObjectConfig>>&,
                    float,
                    const std::unordered_map<int, std::string>&,

                    // FEATURE FLAGS
                    bool,
                    bool,
                    bool,
                    const std::unordered_map<std::string, float>&,

                    // Inventory regeneration
                    unsigned int,

                    // Clipper
                    const std::shared_ptr<ClipperConfig>&>(),
           py::arg("num_agents"),
           py::arg("max_steps"),
           py::arg("episode_truncates"),
           py::arg("obs_width"),
           py::arg("obs_height"),
           py::arg("resource_names"),
           py::arg("num_observation_tokens"),
           py::arg("global_observations") = std::vector<GlobalObsFeature>(),
           py::arg("global_obs"),
           py::arg("actions"),
           py::arg("objects"),
           py::arg("resource_loss_prob") = 0.0f,
           py::arg("tag_id_map") = std::unordered_map<int, std::string>(),

           // FEATURE FLAGS
           py::arg("track_movement_metrics"),
           py::arg("recipe_details_obs") = false,
           py::arg("allow_diagonals") = false,
           py::arg("reward_estimates") = std::unordered_map<std::string, float>(),

           // Inventory regeneration
           py::arg("inventory_regen_interval") = 0,

           // Clipper
           py::arg("clipper") = std::shared_ptr<ClipperConfig>(nullptr))
      .def_readwrite("num_agents", &GameConfig::num_agents)
      .def_readwrite("max_steps", &GameConfig::max_steps)
      .def_readwrite("episode_truncates", &GameConfig::episode_truncates)
      .def_readwrite("obs_width", &GameConfig::obs_width)
      .def_readwrite("obs_height", &GameConfig::obs_height)
      .def_readwrite("resource_names", &GameConfig::resource_names)
      .def_readwrite("num_observation_tokens", &GameConfig::num_observation_tokens)

      // Global observations - NEW flexible system
      .def_readwrite("global_observations", &GameConfig::global_observations)

      // Global observations - OLD struct (deprecated, for backwards compatibility)
      .def_readwrite("global_obs", &GameConfig::global_obs)

      // We don't expose these since they're copied on read, and this means that mutations
      // to the dictionaries don't impact the underlying cpp objects. This is confusing!
      // This can be fixed, but until we do that, we're not exposing these.
      // .def_readwrite("actions", &GameConfig::actions)
      // .def_readwrite("objects", &GameConfig::objects);

      .def_readwrite("resource_loss_prob", &GameConfig::resource_loss_prob)
      .def_readwrite("tag_id_map", &GameConfig::tag_id_map)

      // FEATURE FLAGS
      .def_readwrite("track_movement_metrics", &GameConfig::track_movement_metrics)
      .def_readwrite("recipe_details_obs", &GameConfig::recipe_details_obs)
      .def_readwrite("allow_diagonals", &GameConfig::allow_diagonals)
      .def_readwrite("reward_estimates", &GameConfig::reward_estimates)

      // Inventory regeneration
      .def_readwrite("inventory_regen_interval", &GameConfig::inventory_regen_interval)

      // Clipper
      .def_readwrite("clipper", &GameConfig::clipper);
}

#endif  // PACKAGES_METTAGRID_CPP_INCLUDE_METTAGRID_CONFIG_METTAGRID_CONFIG_HPP_
