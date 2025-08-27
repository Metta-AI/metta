#ifndef OBJECTS_PLANT_HPP_
#define OBJECTS_PLANT_HPP_

#include <string>
#include <vector>

#include "../event.hpp"
#include "../grid.hpp"
#include "../grid_object.hpp"
#include "constants.hpp"

// #MettagridConfig
struct PlantConfig : public GridObjectConfig {
  PlantConfig(TypeId type_id, const std::string& type_name, unsigned int grow_ticks)
      : GridObjectConfig(type_id, type_name), grow_ticks(grow_ticks) {}

  unsigned int grow_ticks;  // ticks between growth events
};

class Plant : public GridObject {
public:
  unsigned int grow_ticks;
  EventManager* event_manager;

  Plant(GridCoord r, GridCoord c, const PlantConfig& cfg) : grow_ticks(cfg.grow_ticks), event_manager(nullptr) {
    GridObject::init(cfg.type_id, cfg.type_name, GridLocation(r, c, GridLayer::ObjectLayer));
  }

  void set_event_manager(EventManager* em) {
    this->event_manager = em;
    // Schedule first growth
    if (this->event_manager && this->id != 0) {
      this->event_manager->schedule_event(EventType::PlantGrow, grow_ticks, this->id, 0);
    }
  }

  // Called by PlantGrowHandler
  void grow_once() {
    // Schedule next growth first to avoid missing cycles
    if (this->event_manager && this->id != 0) {
      this->event_manager->schedule_event(EventType::PlantGrow, grow_ticks, this->id, 0);
    }

    // Try to grow to the north (r-1, same c) if empty
    Grid* grid = this->event_manager ? this->event_manager->grid : nullptr;
    if (!grid) return;

    if (this->location.r == 0) return;

    GridCoord nr = static_cast<GridCoord>(this->location.r - 1);
    GridCoord nc = this->location.c;
    // Only place on ObjectLayer and only if empty at that layer
    if (!grid->is_empty_at_layer(nr, nc, GridLayer::ObjectLayer)) return;

    // Create a new Plant at (nr, nc)
    Plant* child = new Plant(nr, nc, PlantConfig(this->type_id, this->type_name, this->grow_ticks));
    if (grid->add_object(child)) {
      // Wire events for the new child
      child->set_event_manager(this->event_manager);
    } else {
      // If we failed to add, delete to avoid leak
      delete child;
    }
  }

  std::vector<PartialObservationToken> obs_features() const override {
    std::vector<PartialObservationToken> features;
    features.reserve(1);
    features.push_back({ObservationFeature::TypeId, static_cast<ObservationType>(this->type_id)});
    return features;
  }
};

// Event handler for PlantGrow
class PlantGrowHandler : public EventHandler {
public:
  explicit PlantGrowHandler(EventManager* event_manager) : EventHandler(event_manager) {}

  void handle_event(GridObjectId obj_id, EventArg /*arg*/) override {
    Plant* plant = static_cast<Plant*>(this->event_manager->grid->object(obj_id));
    if (!plant) return;
    plant->grow_once();
  }
};

#endif  // OBJECTS_PLANT_HPP_
