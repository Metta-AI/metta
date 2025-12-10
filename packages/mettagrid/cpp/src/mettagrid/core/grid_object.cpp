#include "core/grid_object.hpp"

#include "core/grid.hpp"

void AOEHelper::register_effects(GridCoord r, GridCoord c) {
  if (has_aoe()) {
    _grid->apply_aoe(r, c, _config->range, _config->resource_deltas, true);
    _registered = true;
    _location_r = r;
    _location_c = c;
  }
}

void AOEHelper::unregister_effects() {
  if (_registered && has_aoe()) {
    _grid->apply_aoe(_location_r, _location_c, _config->range, _config->resource_deltas, false);
    _registered = false;
  }
}
