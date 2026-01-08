#ifndef PACKAGES_METTAGRID_CPP_INCLUDE_METTAGRID_OBJECTS_ALIGNABLE_HPP_
#define PACKAGES_METTAGRID_CPP_INCLUDE_METTAGRID_OBJECTS_ALIGNABLE_HPP_

// Forward declarations
class Faction;
class Inventory;

/**
 * Interface for objects that can belong to a Faction group.
 */
class Alignable {
private:
  // Currently limited to one faction, but expected to support multiple in the future.
  Faction* _faction = nullptr;

public:
  virtual ~Alignable() = default;

  // Set the faction this object belongs to
  void setFaction(Faction* faction);

  // Remove this object from its current faction
  void clearFaction();

  // Get the current faction
  Faction* getFaction() const {
    return _faction;
  }

  // Get the inventory of the faction (if any)
  Inventory* faction_inventory() const;
};

#endif  // PACKAGES_METTAGRID_CPP_INCLUDE_METTAGRID_OBJECTS_ALIGNABLE_HPP_
