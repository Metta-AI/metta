#ifndef PACKAGES_METTAGRID_CPP_INCLUDE_METTAGRID_OBJECTS_ALIGNABLE_HPP_
#define PACKAGES_METTAGRID_CPP_INCLUDE_METTAGRID_OBJECTS_ALIGNABLE_HPP_

// Forward declarations
class Collective;
class Inventory;

/**
 * Interface for objects that can belong to a Collective group.
 */
class Alignable {
private:
  // Currently limited to one collective, but expected to support multiple in the future.
  Collective* _collective = nullptr;

public:
  virtual ~Alignable() = default;

  // Set the collective this object belongs to
  void setCollective(Collective* collective);

  // Remove this object from its current collective
  void clearCollective();

  // Get the current collective
  Collective* getCollective() const {
    return _collective;
  }

  // Get the inventory of the collective (if any)
  Inventory* collective_inventory() const;
};

#endif  // PACKAGES_METTAGRID_CPP_INCLUDE_METTAGRID_OBJECTS_ALIGNABLE_HPP_
