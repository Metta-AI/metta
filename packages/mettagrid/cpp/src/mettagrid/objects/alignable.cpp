#include "objects/alignable.hpp"

#include "objects/collective.hpp"

void Alignable::setCollective(Collective* collective) {
  // Remove from old collective if set
  if (_collective != nullptr) {
    _collective->removeMember(this);
  }
  // Set new collective
  _collective = collective;
  // Add to new collective if not null
  if (_collective != nullptr) {
    _collective->addMember(this);
  }
}

void Alignable::clearCollective() {
  if (_collective != nullptr) {
    _collective->removeMember(this);
    _collective = nullptr;
  }
}

Inventory* Alignable::collective_inventory() const {
  if (_collective != nullptr) {
    return &_collective->inventory;
  }
  return nullptr;
}
