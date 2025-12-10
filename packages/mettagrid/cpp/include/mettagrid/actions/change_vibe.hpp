#ifndef PACKAGES_METTAGRID_CPP_INCLUDE_METTAGRID_ACTIONS_CHANGE_VIBE_HPP_
#define PACKAGES_METTAGRID_CPP_INCLUDE_METTAGRID_ACTIONS_CHANGE_VIBE_HPP_

#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include <string>

#include "actions/action_handler.hpp"
#include "core/types.hpp"
#include "objects/agent.hpp"

struct ChangeVibeActionConfig : public ActionConfig {
  const ObservationType number_of_vibes;

  ChangeVibeActionConfig(const std::unordered_map<InventoryItem, InventoryQuantity>& required_resources,
                         const std::unordered_map<InventoryItem, InventoryQuantity>& consumed_resources,
                         const ObservationType number_of_vibes)
      : ActionConfig(required_resources, consumed_resources), number_of_vibes(number_of_vibes) {}
};

// Forward declaration
struct GameConfig;

class ChangeVibe : public ActionHandler {
public:
  explicit ChangeVibe(const ChangeVibeActionConfig& cfg, const GameConfig* game_config)
      : ActionHandler(cfg, "change_vibe"), _number_of_vibes(cfg.number_of_vibes), _game_config(game_config) {}

  std::vector<Action> create_actions() override {
    std::vector<Action> actions;
    for (unsigned char i = 0; i < _number_of_vibes; ++i) {
      std::string action_name;
      if (_game_config && i < _game_config->vibe_names.size()) {
        action_name = "change_vibe_" + _game_config->vibe_names[i];
      } else {
        action_name = "change_vibe_" + std::to_string(i);
      }
      actions.emplace_back(this, action_name, static_cast<ActionArg>(i));
    }
    return actions;
  }

protected:
  const ObservationType _number_of_vibes;
  const GameConfig* _game_config;

  bool _handle_action(Agent& actor, ActionArg arg) override {
    actor.vibe = static_cast<ObservationType>(arg);  // ActionArg is int32 for puffer compatibility
    return true;
  }
};

namespace py = pybind11;

inline void bind_change_vibe_action_config(py::module& m) {
  py::class_<ChangeVibeActionConfig, ActionConfig, std::shared_ptr<ChangeVibeActionConfig>>(m, "ChangeVibeActionConfig")
      .def(py::init<const std::unordered_map<InventoryItem, InventoryQuantity>&,
                    const std::unordered_map<InventoryItem, InventoryQuantity>&,
                    const int>(),
           py::arg("required_resources") = std::unordered_map<InventoryItem, InventoryQuantity>(),
           py::arg("consumed_resources") = std::unordered_map<InventoryItem, InventoryQuantity>(),
           py::arg("number_of_vibes"))
      .def_readonly("number_of_vibes", &ChangeVibeActionConfig::number_of_vibes);
}
#endif  // PACKAGES_METTAGRID_CPP_INCLUDE_METTAGRID_ACTIONS_CHANGE_VIBE_HPP_
