#ifndef OBJECTS_BOX_HPP_
#define OBJECTS_BOX_HPP_

#include <string>

#include "constants.hpp"
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

class Box : public HasInventory {
public:
  GridObjectId creator_agent_object_id;
  unsigned char creator_agent_id;
  std::map<InventoryItem, InventoryQuantity> returned_resources;

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

  std::vector<PartialObservationToken> obs_features() const override {
    std::vector<PartialObservationToken> features;
    features.reserve(2);
    features.push_back({ObservationFeature::TypeId, static_cast<ObservationType>(this->type_id)});
    features.push_back({ObservationFeature::Group, static_cast<ObservationType>(creator_agent_id)});
    return features;
  }
};

#endif  //  OBJECTS_BOX_HPP_
