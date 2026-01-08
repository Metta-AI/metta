#include "objects/alignable.hpp"

#include "objects/faction.hpp"

void Alignable::setFaction(Faction* faction) {
  // Remove from old faction if set
  if (_faction != nullptr) {
    _faction->removeMember(this);
  }
  // Set new faction
  _faction = faction;
  // Add to new faction if not null
  if (_faction != nullptr) {
    _faction->addMember(this);
  }
}

void Alignable::clearFaction() {
  if (_faction != nullptr) {
    _faction->removeMember(this);
    _faction = nullptr;
  }
}

Inventory* Alignable::faction_inventory() const {
  if (_faction != nullptr) {
    return &_faction->inventory;
  }
  return nullptr;
}
