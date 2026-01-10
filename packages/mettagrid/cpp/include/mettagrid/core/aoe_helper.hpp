#ifndef PACKAGES_METTAGRID_CPP_INCLUDE_METTAGRID_CORE_AOE_HELPER_HPP_
#define PACKAGES_METTAGRID_CPP_INCLUDE_METTAGRID_CORE_AOE_HELPER_HPP_

#include <memory>
#include <unordered_map>
#include <vector>

#include "core/grid_object.hpp"
#include "handler/handler.hpp"

namespace mettagrid {

// An AOE handler source registered at a cell location
struct AOEHandlerSource {
  std::shared_ptr<Handler> handler;
  GridObject* source;  // The object that owns this handler

  AOEHandlerSource() : handler(nullptr), source(nullptr) {}
  AOEHandlerSource(std::shared_ptr<Handler> h, GridObject* src) : handler(std::move(h)), source(src) {}
};

/**
 * AOEEffectGrid manages cell-based AOE handler registration.
 *
 * Instead of tracking individual targets, handlers are registered at cell locations.
 * Objects query handlers at their location and apply those that pass filters.
 *
 * Usage:
 *   1. Create AOEEffectGrid with grid dimensions
 *   2. Call register_source() when an AOE source is placed
 *   3. Call unregister_source() when an AOE source is removed
 *   4. Objects call apply_effects_at() to get handlers applied at their location
 */
class AOEEffectGrid {
public:
  AOEEffectGrid(GridCoord height, GridCoord width);
  ~AOEEffectGrid() = default;

  // Register an AOE source - adds handler to all cells within radius
  void register_source(GridObject& source, std::shared_ptr<Handler> handler);

  // Unregister all AOE handlers for a source - removes from all cells
  void unregister_source(GridObject& source);

  // Apply all AOE handlers at a location to a target object
  // Handler filters are checked before applying mutations
  void apply_effects_at(const GridLocation& loc, GridObject& target);

  // Get number of handler sources at a location (for testing/debugging)
  size_t effect_count_at(const GridLocation& loc) const;

private:
  // Hash function for GridLocation
  struct LocationHash {
    size_t operator()(const GridLocation& loc) const {
      return std::hash<GridCoord>()(loc.r) ^ (std::hash<GridCoord>()(loc.c) << 16);
    }
  };

  GridCoord _height;
  GridCoord _width;

  // Map from cell location to list of handler sources affecting that cell
  std::unordered_map<GridLocation, std::vector<AOEHandlerSource>, LocationHash> _cell_effects;

  // Map from source object to its registered handlers (for unregistration)
  std::unordered_map<GridObject*, std::vector<std::shared_ptr<Handler>>> _source_handlers;
};

}  // namespace mettagrid

#endif  // PACKAGES_METTAGRID_CPP_INCLUDE_METTAGRID_CORE_AOE_HELPER_HPP_
