#include "objects/alignable.hpp"

#include "core/grid_object.hpp"
#include "objects/collective.hpp"

void Alignable::setCollective(Collective* collective) {
  // Cast to GridObject* since GridObject inherits from Alignable
  GridObject* self = dynamic_cast<GridObject*>(this);

  // Remove from old collective if set
  if (_collective != nullptr) {
    _collective->removeMember(self);
  }
  // Set new collective
  _collective = collective;
  // Add to new collective if not null
  if (_collective != nullptr) {
    _collective->addMember(self);
  }
}

void Alignable::clearCollective() {
  if (_collective != nullptr) {
    GridObject* self = dynamic_cast<GridObject*>(this);
    _collective->removeMember(self);
    _collective = nullptr;
  }
}

Inventory* Alignable::collective_inventory() const {
  if (_collective != nullptr) {
    return &_collective->inventory;
  }
  return nullptr;
}
