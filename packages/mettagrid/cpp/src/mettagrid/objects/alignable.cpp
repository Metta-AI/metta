#include "objects/alignable.hpp"

#include "core/grid_object.hpp"
#include "objects/faction.hpp"

// Helper to get type_name from Alignable via dynamic_cast to GridObject
static std::string get_alignable_type_name(Alignable* alignable) {
  // All Alignables in practice are also GridObjects (Agent, Assembler, etc.)
  GridObject* grid_obj = dynamic_cast<GridObject*>(alignable);
  if (grid_obj != nullptr) {
    return grid_obj->type_name;
  }
  return "";
}

void Alignable::setFaction(Faction* faction) {
  std::string type_name = get_alignable_type_name(this);

  // Remove from old faction if set
  if (_faction != nullptr) {
    _faction->removeMember(this, type_name);
  }
  // Set new faction
  _faction = faction;
  // Add to new faction if not null
  if (_faction != nullptr) {
    _faction->addMember(this, type_name);
  }
}

void Alignable::clearFaction() {
  if (_faction != nullptr) {
    std::string type_name = get_alignable_type_name(this);
    _faction->removeMember(this, type_name);
    _faction = nullptr;
  }
}

Inventory* Alignable::faction_inventory() const {
  if (_faction != nullptr) {
    return &_faction->inventory;
  }
  return nullptr;
}
