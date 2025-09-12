#ifndef OBJECTS_BOX_HPP_
#define OBJECTS_BOX_HPP_

#include <string>

#include "constants.hpp"
#include "has_inventory.hpp"
#include "inventory_list.hpp"
#include "types.hpp"

// #MettaGridConfig
struct BoxConfig : public GridObjectConfig {
  BoxConfig(TypeId type_id,
            const std::string& type_name,
            const std::map<InventoryItem, InventoryQuantity>& returned_resources)
      : GridObjectConfig(type_id, type_name), returned_resources(returned_resources) {}
  std::map<InventoryItem, InventoryQuantity> returned_resources;
};

class Box : public HasInventory {
private:
  // Inventory management with resource tracking
  InventoryList inventory_list;

public:
  GridObjectId creator_agent_object_id;
  unsigned char creator_agent_id;
  std::map<InventoryItem, InventoryQuantity> returned_resources;

  // Expose inventory and resource_instances for backward compatibility
  std::map<InventoryItem, InventoryQuantity>& inventory = inventory_list.inventory;
  std::map<uint64_t, ResourceInstance>& resource_instances = inventory_list.resource_instances;

  Box(GridCoord r,
      GridCoord c,
      const BoxConfig& config,
      GridObjectId creator_agent_object_id,
      unsigned char creator_agent_id)
      : creator_agent_object_id(creator_agent_object_id),
        creator_agent_id(creator_agent_id),
        returned_resources(config.returned_resources),
        inventory_list(InventoryList()) {  // Box never has resource loss
    GridObject::init(config.type_id, config.type_name, GridLocation(r, c, GridLayer::ObjectLayer));
  }

  ~Box() {}

  // Implement HasInventory interface
  InventoryList& get_inventory_list() override {
    return inventory_list;
  }

  const InventoryList& get_inventory_list() const override {
    return inventory_list;
  }

  // Implement update_inventory using InventoryList with no resource loss
  InventoryDelta update_inventory(InventoryItem item, InventoryDelta delta) override {
    return inventory_list.update_inventory(item, delta, this->id);
  }

  bool is_creator(unsigned char agent_id) const {
    return agent_id == creator_agent_id;
  }

  std::vector<PartialObservationToken> obs_features() const override {
    std::vector<PartialObservationToken> features;
    features.reserve(2);
    features.push_back({ObservationFeature::TypeId, static_cast<ObservationType>(this->type_id)});
    features.push_back({ObservationFeature::Group, static_cast<ObservationType>(creator_agent_id)});
    return features;
  }
};

#endif  //  OBJECTS_BOX_HPP_
