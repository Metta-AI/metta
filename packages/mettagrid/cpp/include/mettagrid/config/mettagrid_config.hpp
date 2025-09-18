#ifndef PACKAGES_METTAGRID_CPP_INCLUDE_METTAGRID_CONFIG_METTAGRID_CONFIG_HPP_
#define PACKAGES_METTAGRID_CPP_INCLUDE_METTAGRID_CONFIG_METTAGRID_CONFIG_HPP_

#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include <map>
#include <memory>
#include <string>
#include <vector>

#include "core/types.hpp"

// Forward declarations
struct ActionConfig;
struct GridObjectConfig;

using ObservationCoord = ObservationType;

struct GlobalObsConfig {
  bool episode_completion_pct = true;
  bool last_action = true;
  bool last_reward = true;
  bool resource_rewards = false;
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
  GlobalObsConfig global_obs;
  std::map<std::string, std::shared_ptr<ActionConfig>> actions;
  std::map<std::string, std::shared_ptr<GridObjectConfig>> objects;
  float resource_loss_prob = 0.0;

  // FEATURE FLAGS
  bool track_movement_metrics = false;
  bool recipe_details_obs = false;
  bool allow_diagonals = false;
  std::map<std::string, float> reward_estimates = {};
};

namespace py = pybind11;

inline void bind_global_obs_config(py::module& m) {
  py::class_<GlobalObsConfig>(m, "GlobalObsConfig")
      .def(py::init<>())
      .def(py::init<bool, bool, bool, bool, bool>(),
           py::arg("episode_completion_pct") = true,
           py::arg("last_action") = true,
           py::arg("last_reward") = true,
           py::arg("resource_rewards") = false,
           py::arg("visitation_counts") = false)
      .def_readwrite("episode_completion_pct", &GlobalObsConfig::episode_completion_pct)
      .def_readwrite("last_action", &GlobalObsConfig::last_action)
      .def_readwrite("last_reward", &GlobalObsConfig::last_reward)
      .def_readwrite("resource_rewards", &GlobalObsConfig::resource_rewards)
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
                    const GlobalObsConfig&,
                    const std::map<std::string, std::shared_ptr<ActionConfig>>&,
                    const std::map<std::string, std::shared_ptr<GridObjectConfig>>&,
                    float,

                    // FEATURE FLAGS
                    bool,
                    bool,
                    bool,
                    const std::map<std::string, float>&>(),
           py::arg("num_agents"),
           py::arg("max_steps"),
           py::arg("episode_truncates"),
           py::arg("obs_width"),
           py::arg("obs_height"),
           py::arg("resource_names"),
           py::arg("num_observation_tokens"),
           py::arg("global_obs"),
           py::arg("actions"),
           py::arg("objects"),
           py::arg("resource_loss_prob") = 0.0f,

           // FEATURE FLAGS
           py::arg("track_movement_metrics"),
           py::arg("recipe_details_obs") = false,
           py::arg("allow_diagonals") = false,
           py::arg("reward_estimates") = std::map<std::string, float>())
      .def_readwrite("num_agents", &GameConfig::num_agents)
      .def_readwrite("max_steps", &GameConfig::max_steps)
      .def_readwrite("episode_truncates", &GameConfig::episode_truncates)
      .def_readwrite("obs_width", &GameConfig::obs_width)
      .def_readwrite("obs_height", &GameConfig::obs_height)
      .def_readwrite("resource_names", &GameConfig::resource_names)
      .def_readwrite("num_observation_tokens", &GameConfig::num_observation_tokens)
      .def_readwrite("global_obs", &GameConfig::global_obs)

      // We don't expose these since they're copied on read, and this means that mutations
      // to the dictionaries don't impact the underlying cpp objects. This is confusing!
      // This can be fixed, but until we do that, we're not exposing these.
      // .def_readwrite("actions", &GameConfig::actions)
      // .def_readwrite("objects", &GameConfig::objects);

      .def_readwrite("resource_loss_prob", &GameConfig::resource_loss_prob)

      // FEATURE FLAGS
      .def_readwrite("track_movement_metrics", &GameConfig::track_movement_metrics)
      .def_readwrite("recipe_details_obs", &GameConfig::recipe_details_obs)
      .def_readwrite("allow_diagonals", &GameConfig::allow_diagonals)
      .def_readwrite("reward_estimates", &GameConfig::reward_estimates);
}

#endif  // PACKAGES_METTAGRID_CPP_INCLUDE_METTAGRID_CONFIG_METTAGRID_CONFIG_HPP_
