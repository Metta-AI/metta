#ifndef OBJECTS_BOX_HPP_
#define OBJECTS_BOX_HPP_

#include <string>
#include "has_inventory.hpp"
#include "constants.hpp"
#include "types.hpp"

// #MettagridConfig
struct BoxConfig : public GridObjectConfig {
    BoxConfig(TypeId type_id, const std::string& type_name)
        : GridObjectConfig(type_id, type_name) {}
  };

class Box : public HasInventory {
public:
    GridObjectId creator_agent_object_id;
    unsigned char creator_agent_id;

    Box(GridCoord r, GridCoord c, TypeId type_id, const std::string& type_name,
        GridObjectId creator_agent_object_id, unsigned char creator_agent_id,
        InventoryItem blue_battery_item)
        : creator_agent_object_id(creator_agent_object_id),
          creator_agent_id(creator_agent_id) {
        GridObject::init(type_id, type_name, GridLocation(r, c, GridLayer::ObjectLayer));
        // Add one blue battery to inventory
        this->inventory[blue_battery_item] = 1;
    }

    ~Box() {
    }

    bool is_creator(unsigned char agent_id) const {
        return agent_id == creator_agent_id;
    }

    std::vector<PartialObservationToken> obs_features() const override {
        std::vector<PartialObservationToken> features;
        features.reserve(3);
        features.push_back({ObservationFeature::TypeId, static_cast<ObservationType>(this->type_id)});
        features.push_back({ObservationFeature::Group, static_cast<ObservationType>(creator_agent_id)});
        // Optionally, add inventory info if needed
        for (const auto& [item, amount] : this->inventory) {
            if (amount > 0) {
                features.push_back({static_cast<ObservationType>(InventoryFeatureOffset + item),
                    static_cast<ObservationType>(amount)});
            }
        }
        return features;
    }
};

#endif // OBJECTS_BOX_HPP_
