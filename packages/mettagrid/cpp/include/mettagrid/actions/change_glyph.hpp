#ifndef PACKAGES_METTAGRID_CPP_INCLUDE_METTAGRID_ACTIONS_CHANGE_GLYPH_HPP_
#define PACKAGES_METTAGRID_CPP_INCLUDE_METTAGRID_ACTIONS_CHANGE_GLYPH_HPP_

#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include <string>

#include "actions/action_handler.hpp"
#include "core/types.hpp"
#include "objects/agent.hpp"

struct ChangeGlyphActionConfig : public ActionConfig {
  const ObservationType number_of_glyphs;

  ChangeGlyphActionConfig(const std::unordered_map<InventoryItem, InventoryQuantity>& required_resources,
                          const std::unordered_map<InventoryItem, InventoryProbability>& consumed_resources,
                          const ObservationType number_of_glyphs)
      : ActionConfig(required_resources, consumed_resources), number_of_glyphs(number_of_glyphs) {}
};

// Forward declaration
struct GameConfig;

class ChangeGlyph : public ActionHandler {
public:
  explicit ChangeGlyph(const ChangeGlyphActionConfig& cfg, const GameConfig* game_config)
      : ActionHandler(cfg, "change_glyph"), _number_of_glyphs(cfg.number_of_glyphs), _game_config(game_config) {}

  std::vector<Action> create_actions() override {
    std::vector<Action> actions;
    unsigned char max_arg = _number_of_glyphs > 0 ? _number_of_glyphs - 1 : 0;
    for (unsigned char i = 0; i <= max_arg; ++i) {
      std::string action_name;
      if (_game_config && i < _game_config->vibe_names.size()) {
        action_name = "change_glyph_" + _game_config->vibe_names[i];
      } else {
        action_name = "change_glyph_" + std::to_string(i);
      }
      actions.emplace_back(this, action_name, static_cast<ActionArg>(i));
    }
    return actions;
  }

protected:
  const ObservationType _number_of_glyphs;
  const GameConfig* _game_config;

  bool _handle_action(Agent& actor, ActionArg arg) override {
    actor.glyph = static_cast<ObservationType>(arg);  // ActionArg is int32 for puffer compatibility
    return true;
  }
};

namespace py = pybind11;

inline void bind_change_glyph_action_config(py::module& m) {
  py::class_<ChangeGlyphActionConfig, ActionConfig, std::shared_ptr<ChangeGlyphActionConfig>>(m,
                                                                                              "ChangeGlyphActionConfig")
      .def(py::init<const std::unordered_map<InventoryItem, InventoryQuantity>&,
                    const std::unordered_map<InventoryItem, InventoryProbability>&,
                    const int>(),
           py::arg("required_resources") = std::unordered_map<InventoryItem, InventoryQuantity>(),
           py::arg("consumed_resources") = std::unordered_map<InventoryItem, InventoryProbability>(),
           py::arg("number_of_glyphs"))
      .def_readonly("number_of_glyphs", &ChangeGlyphActionConfig::number_of_glyphs);
}
#endif  // PACKAGES_METTAGRID_CPP_INCLUDE_METTAGRID_ACTIONS_CHANGE_GLYPH_HPP_
