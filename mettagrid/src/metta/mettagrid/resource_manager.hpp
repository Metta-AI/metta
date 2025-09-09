#ifndef RESOURCE_MANAGER_HPP_
#define RESOURCE_MANAGER_HPP_

#include <map>
#include <random>
#include <vector>

#include "grid.hpp"
#include "objects/has_inventory.hpp"
#include "types.hpp"

// Forward declarations
class Grid;
class HasInventory;

class ResourceManager {
public:
  ResourceManager(Grid* grid, std::mt19937& rng);

  // Main processing method called once per step
  void step();

  // Object registration and tracking
  void register_inventory_object(HasInventory* obj, const std::string& group_name = "");
  void register_agent(HasInventory* obj, const std::string& group_name);
  void register_object(HasInventory* obj);
  void unregister_inventory_object(GridObjectId object_id);


  // Inventory change callback
  void on_inventory_changed(GridObjectId object_id, InventoryItem item, InventoryDelta delta);

private:
  Grid* _grid;
  std::mt19937& _rng;

  // Group-based tracking maps (group_name -> list of HasInventory objects)
  std::map<std::string, std::vector<HasInventory*>> _objects_by_group;

  // Cached group names for objects (object_id -> group_name)
  std::map<GridObjectId, std::string> _object_group_names;

  // Group-based inventory tracking maps
  // Maps: group_name -> (item_type -> total_quantity)
  std::map<std::string, std::map<InventoryItem, InventoryQuantity>> _group_inventory_totals;

  // Cached bin data for efficient step() processing
  // Maps: (group_name, item) -> vector of (object, quantity)
  std::map<std::pair<std::string, InventoryItem>, std::vector<std::pair<HasInventory*, InventoryQuantity>>> _bins;

  // Cached loss probabilities for each bin
  // Maps: (group_name, item) -> loss_probability
  std::map<std::pair<std::string, InventoryItem>, float> _bin_loss_probabilities;

  // Helper methods
  HasInventory* _get_inventory_object(GridObjectId object_id) const;
  std::string _get_group_name(HasInventory* obj) const;
  void _validate_object_id(GridObjectId id) const;
  void _update_group_maps();
  void _update_inventory_totals();
  void _update_bins();
  void _add_object_to_bins(HasInventory* obj, const std::string& group_name);
  void _remove_object_from_bins(GridObjectId object_id, const std::string& group_name);
  void _update_object_in_bins(HasInventory* obj, const std::string& group_name, InventoryItem item, InventoryDelta delta);
};

#endif  // RESOURCE_MANAGER_HPP_
