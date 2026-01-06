#ifndef PACKAGES_METTAGRID_CPP_INCLUDE_METTAGRID_OBJECTS_INVENTORY_HPP_
#define PACKAGES_METTAGRID_CPP_INCLUDE_METTAGRID_OBJECTS_INVENTORY_HPP_

#include <limits>
#include <string>
#include <unordered_map>
#include <vector>

#include "core/types.hpp"
#include "objects/constants.hpp"
#include "objects/inventory_config.hpp"

class HasInventory;

struct SharedInventoryLimit {
  InventoryQuantity base_limit;
  // Modifiers: item_id -> bonus_per_item
  std::unordered_map<InventoryItem, InventoryQuantity> modifiers;
  // How much do we have of whatever-this-limit-applies-to
  InventoryQuantity amount;

  // Get the effective limit (base + sum of modifier bonuses)
  InventoryQuantity effective_limit(const std::unordered_map<InventoryItem, InventoryQuantity>& inventory) const {
    int effective = base_limit;
    for (const auto& [item, bonus] : modifiers) {
      auto it = inventory.find(item);
      if (it != inventory.end()) {
        effective += static_cast<int>(it->second) * static_cast<int>(bonus);
      }
    }
    // Clamp to valid range (0 to max InventoryQuantity which is uint16_t)
    if (effective < 0) effective = 0;
    if (effective > 65535) effective = 65535;
    return static_cast<InventoryQuantity>(effective);
  }
};

class Inventory {
private:
  std::unordered_map<InventoryItem, InventoryQuantity> _inventory;
  std::unordered_map<InventoryItem, SharedInventoryLimit*> _limits;
  // The HasInventory that owns this inventory. If we want multiple things to react to changes,
  // we can make this a vector.
  HasInventory* _owner;

public:
  // Constructor and Destructor
  explicit Inventory(const InventoryConfig& cfg,
                     HasInventory* owner = nullptr,
                     const std::vector<std::string>* resource_names = nullptr,
                     const std::unordered_map<std::string, ObservationType>* feature_ids = nullptr);
  ~Inventory();

  // Update the inventory for a specific item
  // If ignore_limits is true, the update will bypass limit checks (used for initial inventory)
  InventoryDelta update(InventoryItem item, InventoryDelta attempted_delta, bool ignore_limits = false);

  // Get the amount of a specific item
  InventoryQuantity amount(InventoryItem item) const;

  // Get the free space for a specific item
  InventoryQuantity free_space(InventoryItem item) const;

  // Get all inventory items
  std::unordered_map<InventoryItem, InventoryQuantity> get() const;

  // Enforce all limits - drop excess items when limits decrease (e.g., after losing gear modifiers)
  void enforce_all_limits();

  // Check if an item is a modifier for any limit
  bool is_modifier(InventoryItem item) const;
};

#endif  // PACKAGES_METTAGRID_CPP_INCLUDE_METTAGRID_OBJECTS_INVENTORY_HPP_
