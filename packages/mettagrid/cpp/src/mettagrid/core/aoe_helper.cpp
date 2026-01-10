#include "core/aoe_helper.hpp"

#include <algorithm>

#include "handler/handler_context.hpp"
#include "objects/has_inventory.hpp"

namespace mettagrid {

AOEEffectGrid::AOEEffectGrid(GridCoord height, GridCoord width) : _height(height), _width(width) {}

void AOEEffectGrid::register_source(GridObject& source, std::shared_ptr<Handler> handler) {
  // Store the handler for later unregistration
  _source_handlers[&source].push_back(handler);

  const GridLocation& source_loc = source.location;
  int radius = handler->radius();
  AOEHandlerSource handler_source(handler, &source);

  // Register handler at all cells within L-infinity (Chebyshev) distance
  for (int dr = -radius; dr <= radius; ++dr) {
    int cell_r = static_cast<int>(source_loc.r) + dr;
    if (cell_r < 0 || cell_r >= static_cast<int>(_height)) {
      continue;
    }
    for (int dc = -radius; dc <= radius; ++dc) {
      int cell_c = static_cast<int>(source_loc.c) + dc;
      if (cell_c < 0 || cell_c >= static_cast<int>(_width)) {
        continue;
      }
      GridLocation cell_loc(static_cast<GridCoord>(cell_r), static_cast<GridCoord>(cell_c));
      _cell_effects[cell_loc].push_back(handler_source);
    }
  }
}

void AOEEffectGrid::unregister_source(GridObject& source) {
  auto handlers_it = _source_handlers.find(&source);
  if (handlers_it == _source_handlers.end()) {
    return;
  }

  // Get the maximum radius from all handlers for this source
  int max_radius = 0;
  for (const auto& handler : handlers_it->second) {
    max_radius = std::max(max_radius, handler->radius());
  }

  const GridLocation& source_loc = source.location;

  // Remove all handlers from this source from all cells within max radius
  for (int dr = -max_radius; dr <= max_radius; ++dr) {
    int cell_r = static_cast<int>(source_loc.r) + dr;
    if (cell_r < 0 || cell_r >= static_cast<int>(_height)) {
      continue;
    }
    for (int dc = -max_radius; dc <= max_radius; ++dc) {
      int cell_c = static_cast<int>(source_loc.c) + dc;
      if (cell_c < 0 || cell_c >= static_cast<int>(_width)) {
        continue;
      }
      GridLocation cell_loc(static_cast<GridCoord>(cell_r), static_cast<GridCoord>(cell_c));
      auto cell_it = _cell_effects.find(cell_loc);
      if (cell_it != _cell_effects.end()) {
        auto& effects = cell_it->second;
        effects.erase(
            std::remove_if(
                effects.begin(), effects.end(), [&source](const AOEHandlerSource& e) { return e.source == &source; }),
            effects.end());
        if (effects.empty()) {
          _cell_effects.erase(cell_it);
        }
      }
    }
  }

  _source_handlers.erase(handlers_it);
}

void AOEEffectGrid::apply_effects_at(const GridLocation& loc, GridObject& target) {
  auto cell_it = _cell_effects.find(loc);
  if (cell_it == _cell_effects.end()) {
    return;
  }

  for (AOEHandlerSource& effect : cell_it->second) {
    // Skip if target is the source
    if (effect.source == &target) {
      continue;
    }

    // Create context: actor=source, target=affected object
    HandlerContext ctx(effect.source, &target);

    // Handler's filters and mutations are applied via try_apply
    effect.handler->try_apply(ctx);
  }
}

size_t AOEEffectGrid::effect_count_at(const GridLocation& loc) const {
  auto cell_it = _cell_effects.find(loc);
  if (cell_it == _cell_effects.end()) {
    return 0;
  }
  return cell_it->second.size();
}

}  // namespace mettagrid
