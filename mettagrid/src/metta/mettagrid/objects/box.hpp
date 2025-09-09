#ifndef OBJECTS_BOX_HPP_
#define OBJECTS_BOX_HPP_

#include <algorithm>
#include <string>

#include "constants.hpp"
#include "grid_object.hpp"
#include "has_inventory.hpp"
#include "types.hpp"

// #MettaGridConfig
struct BoxConfig : public GridObjectConfig {
  BoxConfig(TypeId type_id,
            const std::string& type_name,
            const std::map<InventoryItem, InventoryQuantity>& returned_resources)
      : GridObjectConfig(type_id, type_name), returned_resources(returned_resources) {}
  std::map<InventoryItem, InventoryQuantity> returned_resources;
};

class Box : public GridObject, public virtual HasInventory {
public:
  GridObjectId creator_agent_object_id;
  unsigned char creator_agent_id;
  std::map<InventoryItem, InventoryQuantity> returned_resources;

  // HasInventory interface implementation
  std::map<InventoryItem, InventoryQuantity> inventory;
  HasInventory::InventoryChangeCallback inventory_callback;

  Box(GridCoord r,
      GridCoord c,
      const BoxConfig& config,
      GridObjectId creator_agent_object_id,
      unsigned char creator_agent_id)
      : creator_agent_object_id(creator_agent_object_id),
        creator_agent_id(creator_agent_id),
        returned_resources(config.returned_resources) {
    GridObject::init(config.type_id, config.type_name, GridLocation(r, c, GridLayer::ObjectLayer));
  }

  ~Box() {}

  bool is_creator(unsigned char agent_id) const {
    return agent_id == creator_agent_id;
  }

  // HasInventory interface implementation
  const std::map<InventoryItem, InventoryQuantity>& get_inventory() const override {
    return inventory;
  }

  InventoryDelta update_inventory(InventoryItem item, InventoryDelta delta) override {
    InventoryQuantity initial_amount = this->inventory[item];
    int new_amount = static_cast<int>(initial_amount + delta);

    constexpr int min = std::numeric_limits<InventoryQuantity>::min();
    constexpr int max = std::numeric_limits<InventoryQuantity>::max();
    InventoryQuantity clamped_amount = static_cast<InventoryQuantity>(std::clamp(new_amount, min, max));

    if (clamped_amount == 0) {
      this->inventory.erase(item);
    } else {
      this->inventory[item] = clamped_amount;
    }

    InventoryDelta clamped_delta = clamped_amount - initial_amount;

    // Call callback if inventory actually changed
    if (clamped_delta != 0 && inventory_callback) {
      inventory_callback(this->id, item, clamped_delta);
    }

    return clamped_delta;
  }


  void set_inventory_callback(HasInventory::InventoryChangeCallback callback) override {
    inventory_callback = callback;
  }

  // Resource loss probability method
  const std::map<InventoryItem, float>& get_resource_loss_prob() const override {
    static const std::map<InventoryItem, float> empty_map;
    return empty_map;
  }

  // Type name method
  const std::string& type_name() const override {
    return GridObject::type_name;
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
