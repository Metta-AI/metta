#ifndef ACTIONS_CHANGE_GLYPH_HPP_
#define ACTIONS_CHANGE_GLYPH_HPP_

#include <string>

#include "action_handler.hpp"
#include "objects/agent.hpp"
#include "types.hpp"

struct ChangeGlyphActionConfig : public ActionConfig {
  const ObservationType number_of_glyphs;

  ChangeGlyphActionConfig(const std::map<InventoryItem, InventoryQuantity>& required_resources,
                          const std::map<InventoryItem, InventoryQuantity>& consumed_resources,
                          const ObservationType number_of_glyphs)
      : ActionConfig(required_resources, consumed_resources), number_of_glyphs(number_of_glyphs) {}
};

class ChangeGlyph : public ActionHandler {
public:
  explicit ChangeGlyph(const ChangeGlyphActionConfig& cfg)
      : ActionHandler(cfg, "change_glyph"), _number_of_glyphs(cfg.number_of_glyphs) {}

  unsigned char max_arg() const override {
    // Return number_of_glyphs - 1 since args are 0-indexed
    return _number_of_glyphs > 0 ? _number_of_glyphs - 1 : 0;
  }

protected:
  const ObservationType _number_of_glyphs;

  bool _handle_action(Agent* actor, ActionArg arg) override {
    actor->glyph = arg;
    return true;
  }
};

#endif  // ACTIONS_CHANGE_GLYPH_HPP_
