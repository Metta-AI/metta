#include "core/grid_object.hpp"

#include "core/grid.hpp"

// Commons support has been moved to Alignable interface

void AOEHelper::register_effects(GridCoord r, GridCoord c) {
  if (has_aoe()) {
    // Register as individual source for commons-aware filtering
    _grid->register_aoe_source(r, c, _config->range, _owner, _config);
    // Also apply to legacy accumulated effects for backward compatibility
    _grid->apply_aoe(r, c, _config->range, _config->resource_deltas, true);
    _registered = true;
    _location_r = r;
    _location_c = c;
  }
}

void AOEHelper::unregister_effects() {
  if (_registered && has_aoe()) {
    // Unregister individual source
    _grid->unregister_aoe_source(_location_r, _location_c, _config->range, _owner);
    // Also remove from legacy accumulated effects
    _grid->apply_aoe(_location_r, _location_c, _config->range, _config->resource_deltas, false);
    _registered = false;
  }
}
