#ifndef OBJECTS_HAS_INVENTORY_HPP_
#define OBJECTS_HAS_INVENTORY_HPP_

#include <functional>
#include <map>

#include "constants.hpp"
#include "types.hpp"

// Forward declaration
class GridObject;

class HasInventory {
public:
  virtual ~HasInventory() = default;

  // Callback function type for inventory changes
  using InventoryChangeCallback = std::function<void(GridObjectId, InventoryItem, InventoryDelta)>;

  // Pure virtual methods
  virtual const std::map<InventoryItem, InventoryQuantity>& get_inventory() const = 0;
  virtual InventoryDelta update_inventory(InventoryItem item, InventoryDelta delta) = 0;
  virtual void set_inventory_callback(InventoryChangeCallback callback) = 0;

  // Resource loss probability method
  virtual const std::map<InventoryItem, float>& get_resource_loss_prob() const = 0;
};

#endif  // OBJECTS_HAS_INVENTORY_HPP_
