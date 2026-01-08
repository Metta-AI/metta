#include "core/grid_object.hpp"

#include "core/grid.hpp"

void AOEHelper::register_effects(GridCoord r, GridCoord c) {
  if (has_aoe()) {
    _grid->register_aoe_source(r, c, _config->range, _owner, _config);
    _registered = true;
    _location_r = r;
    _location_c = c;
    _grid->register_aoe_helper(this);
  }
}

void AOEHelper::unregister_effects() {
  if (_registered && has_aoe()) {
    _grid->unregister_aoe_source(_location_r, _location_c, _config->range, _owner);
    _grid->unregister_aoe_helper(this);
    _registered = false;
  }
}
