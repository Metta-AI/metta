#include "core/grid_object.hpp"

#include "objects/commons.hpp"

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
