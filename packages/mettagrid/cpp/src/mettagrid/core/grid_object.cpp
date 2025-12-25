#include "core/grid_object.hpp"

#include "core/grid.hpp"
#include "objects/commons.hpp"

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

void GridObject::setCommons(Commons* commons) {
  // Remove from old commons if set
  if (_commons != nullptr) {
    _commons->removeMember(this);
  }
  // Set new commons
  _commons = commons;
  // Add to new commons if not null
  if (_commons != nullptr) {
    _commons->addMember(this);
  }
}

void GridObject::clearCommons() {
  if (_commons != nullptr) {
    _commons->removeMember(this);
    _commons = nullptr;
  }
}

Inventory* GridObject::commons_inventory() const {
  if (_commons != nullptr) {
    return &_commons->inventory;
  }
  return nullptr;
}
