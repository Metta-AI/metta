#include "core/aoe_helper.hpp"

#include <algorithm>

#include "objects/alignable.hpp"
#include "objects/collective.hpp"
#include "objects/has_inventory.hpp"

namespace mettagrid {

AOEEffectGrid::AOEEffectGrid(GridCoord height, GridCoord width) : _height(height), _width(width) {}

void AOEEffectGrid::register_source(const GridObject& source, const AOEConfig& config) {
  // Store the config for later unregistration
  _source_configs[&source] = config;

  const GridLocation& source_loc = source.location;
  int radius = config.radius;
  AOEEffectSource effect_source(config, &source);

  // Register effect at all cells within L-infinity (Chebyshev) distance
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
      _cell_effects[cell_loc].push_back(effect_source);
    }
  }
}

void AOEEffectGrid::unregister_source(const GridObject& source) {
  auto config_it = _source_configs.find(&source);
  if (config_it == _source_configs.end()) {
    return;
  }

  const AOEConfig& config = config_it->second;
  const GridLocation& source_loc = source.location;
  int radius = config.radius;

  // Remove effect from all cells within L-infinity (Chebyshev) distance
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
      auto cell_it = _cell_effects.find(cell_loc);
      if (cell_it != _cell_effects.end()) {
        auto& effects = cell_it->second;
        effects.erase(
            std::remove_if(
                effects.begin(), effects.end(), [&source](const AOEEffectSource& e) { return e.source == &source; }),
            effects.end());
        if (effects.empty()) {
          _cell_effects.erase(cell_it);
        }
      }
    }
  }

  _source_configs.erase(config_it);
}

bool AOEEffectGrid::passes_tag_filter(const AOEConfig& config, const GridObject& target) {
  if (config.target_tag_ids.empty()) {
    return true;
  }

  for (int required_tag_id : config.target_tag_ids) {
    for (int target_tag_id : target.tag_ids) {
      if (required_tag_id == target_tag_id) {
        return true;
      }
    }
  }

  return false;
}

bool AOEEffectGrid::passes_alignment_filter(const AOEConfig& config,
                                            const GridObject& source,
                                            const GridObject& target) {
  if (config.alignment_filter == AOEAlignmentFilter::any) {
    return true;
  }

  const Alignable* source_alignable = dynamic_cast<const Alignable*>(&source);
  const Alignable* target_alignable = dynamic_cast<const Alignable*>(&target);

  if (source_alignable == nullptr || target_alignable == nullptr) {
    return false;
  }

  Collective* source_collective = source_alignable->getCollective();
  Collective* target_collective = target_alignable->getCollective();

  // Both must have collectives for collective-based filtering
  if (source_collective == nullptr || target_collective == nullptr) {
    return false;
  }

  bool same = (source_collective == target_collective);

  switch (config.alignment_filter) {
    case AOEAlignmentFilter::same_collective:
      return same;
    case AOEAlignmentFilter::different_collective:
      return !same;
    default:
      return true;
  }
}

void AOEEffectGrid::apply_effects_at(const GridLocation& loc, GridObject& target) {
  auto cell_it = _cell_effects.find(loc);
  if (cell_it == _cell_effects.end()) {
    return;
  }

  HasInventory* target_inventory = dynamic_cast<HasInventory*>(&target);
  if (target_inventory == nullptr) {
    return;
  }

  for (const AOEEffectSource& effect : cell_it->second) {
    // Skip if target is the source
    if (effect.source == &target) {
      continue;
    }

    // Check filters
    if (!passes_tag_filter(effect.config, target)) {
      continue;
    }
    if (!passes_alignment_filter(effect.config, *effect.source, target)) {
      continue;
    }

    // Apply all resource deltas
    for (const AOEResourceDelta& delta : effect.config.deltas) {
      target_inventory->inventory.update(delta.resource_id, delta.delta);
    }
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
