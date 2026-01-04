#include "objects/alignable.hpp"

#include "core/grid_object.hpp"
#include "objects/commons.hpp"

// Helper to get type_name from Alignable via dynamic_cast to GridObject
static std::string get_alignable_type_name(Alignable* alignable) {
  // All Alignables in practice are also GridObjects (Agent, Assembler, CommonsChest)
  GridObject* grid_obj = dynamic_cast<GridObject*>(alignable);
  if (grid_obj != nullptr) {
    return grid_obj->type_name;
  }
  return "";
}

void Alignable::setCommons(Commons* commons) {
  std::string type_name = get_alignable_type_name(this);

  // Remove from old commons if set
  if (_commons != nullptr) {
    _commons->removeMember(this, type_name);
  }
  // Set new commons
  _commons = commons;
  // Add to new commons if not null
  if (_commons != nullptr) {
    _commons->addMember(this, type_name);
  }
}

void Alignable::clearCommons() {
  if (_commons != nullptr) {
    std::string type_name = get_alignable_type_name(this);
    _commons->removeMember(this, type_name);
    _commons = nullptr;
  }
}

Inventory* Alignable::commons_inventory() const {
  if (_commons != nullptr) {
    return &_commons->inventory;
  }
  return nullptr;
}
