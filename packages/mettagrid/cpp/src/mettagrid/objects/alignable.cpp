#include "objects/alignable.hpp"

#include "objects/commons.hpp"

void Alignable::setCommons(Commons* commons) {
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

void Alignable::clearCommons() {
  if (_commons != nullptr) {
    _commons->removeMember(this);
    _commons = nullptr;
  }
}

Inventory* Alignable::commons_inventory() const {
  if (_commons != nullptr) {
    return &_commons->inventory;
  }
  return nullptr;
}
