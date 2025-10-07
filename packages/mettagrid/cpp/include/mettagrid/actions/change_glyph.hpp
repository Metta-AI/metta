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

  ChangeGlyphActionConfig(const std::map<InventoryItem, InventoryQuantity>& required_resources,
                          const std::map<InventoryItem, InventoryProbability>& consumed_resources,
                          const ObservationType number_of_glyphs)
      : ActionConfig(required_resources, consumed_resources), number_of_glyphs(number_of_glyphs) {}
};

class ChangeGlyph : public ActionHandler {
public:
  ChangeGlyph(const ChangeGlyphActionConfig& cfg, ObservationType glyph_index, const std::string& name)
      : ActionHandler(cfg, name), _glyph_index(glyph_index) {}

protected:
  ObservationType _glyph_index;

  bool _handle_action(Agent& actor) override {
    actor.glyph = _glyph_index;
    return true;
  }
};

namespace py = pybind11;

inline void bind_change_glyph_action_config(py::module& m) {
  py::class_<ChangeGlyphActionConfig, ActionConfig, std::shared_ptr<ChangeGlyphActionConfig>>(m,
                                                                                              "ChangeGlyphActionConfig")
      .def(py::init<const std::map<InventoryItem, InventoryQuantity>&,
                    const std::map<InventoryItem, InventoryProbability>&,
                    const int>(),
           py::arg("required_resources") = std::map<InventoryItem, InventoryQuantity>(),
           py::arg("consumed_resources") = std::map<InventoryItem, InventoryProbability>(),
           py::arg("number_of_glyphs"))
      .def_readonly("number_of_glyphs", &ChangeGlyphActionConfig::number_of_glyphs);
}
#endif  // PACKAGES_METTAGRID_CPP_INCLUDE_METTAGRID_ACTIONS_CHANGE_GLYPH_HPP_
