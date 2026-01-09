#ifndef PACKAGES_METTAGRID_CPP_INCLUDE_METTAGRID_CORE_AOE_HELPER_HPP_
#define PACKAGES_METTAGRID_CPP_INCLUDE_METTAGRID_CORE_AOE_HELPER_HPP_

#include <unordered_map>
#include <vector>

#include "core/aoe_config.hpp"
#include "core/grid_object.hpp"

// Forward declarations
class HasInventory;

namespace mettagrid {

// An AOE effect source registered at a cell location
struct AOEEffectSource {
  AOEConfig config;
  const GridObject* source;  // The object that created this effect

  AOEEffectSource() : source(nullptr) {}
  AOEEffectSource(const AOEConfig& cfg, const GridObject* src) : config(cfg), source(src) {}
};

/**
 * AOEEffectGrid manages cell-based AOE effect registration.
 *
 * Instead of tracking individual targets, effects are registered at cell locations.
 * Objects query effects at their location and apply those that pass filters.
 *
 * Usage:
 *   1. Create AOEEffectGrid with grid dimensions
 *   2. Call register_source() when an AOE source is placed
 *   3. Call unregister_source() when an AOE source is removed
 *   4. Objects call apply_effects_at() to get effects applied at their location
 */
class AOEEffectGrid {
public:
  AOEEffectGrid(GridCoord height, GridCoord width);
  ~AOEEffectGrid() = default;

  // Register an AOE source - adds effect to all cells within radius
  void register_source(const GridObject& source, const AOEConfig& config);

  // Unregister an AOE source - removes effect from all cells
  void unregister_source(const GridObject& source);

  // Apply all AOE effects at a location to a target object
  // Filters are checked before applying each effect
  void apply_effects_at(const GridLocation& loc, GridObject& target);

  // Get number of effect sources at a location (for testing/debugging)
  size_t effect_count_at(const GridLocation& loc) const;

private:
  // Check if target passes tag filter for an effect
  static bool passes_tag_filter(const AOEConfig& config, const GridObject& target);

  // Check if target passes alignment filter for an effect
  static bool passes_alignment_filter(const AOEConfig& config, const GridObject& source, const GridObject& target);

  // Hash function for GridLocation
  struct LocationHash {
    size_t operator()(const GridLocation& loc) const {
      return std::hash<GridCoord>()(loc.r) ^ (std::hash<GridCoord>()(loc.c) << 16);
    }
  };

  GridCoord _height;
  GridCoord _width;

  // Map from cell location to list of effect sources affecting that cell
  std::unordered_map<GridLocation, std::vector<AOEEffectSource>, LocationHash> _cell_effects;

  // Map from source object to its registered config (for unregistration)
  std::unordered_map<const GridObject*, AOEConfig> _source_configs;
};

}  // namespace mettagrid

#endif  // PACKAGES_METTAGRID_CPP_INCLUDE_METTAGRID_CORE_AOE_HELPER_HPP_
