#ifndef PACKAGES_METTAGRID_CPP_INCLUDE_METTAGRID_OBJECTS_ALIGNABLE_HPP_
#define PACKAGES_METTAGRID_CPP_INCLUDE_METTAGRID_OBJECTS_ALIGNABLE_HPP_

// Forward declarations
class Commons;
class Inventory;

/**
 * Interface for objects that can belong to a Commons group.
 * Objects that implement this interface can have their commons changed
 * via the transfer action's align functionality.
 */
class Alignable {
private:
  Commons* _commons = nullptr;

public:
  virtual ~Alignable() = default;

  // Set the commons this object belongs to
  void setCommons(Commons* commons);

  // Remove this object from its current commons
  void clearCommons();

  // Get the current commons
  Commons* getCommons() const {
    return _commons;
  }

  // Get the inventory of the commons (if any)
  Inventory* commons_inventory() const;
};

#endif  // PACKAGES_METTAGRID_CPP_INCLUDE_METTAGRID_OBJECTS_ALIGNABLE_HPP_
