#ifndef METTAGRID_METTAGRID_OBJECTS_FREEZE_TOWER_HPP_
#define METTAGRID_METTAGRID_OBJECTS_FREEZE_TOWER_HPP_

#include <cassert>
#include <string>
#include <vector>

#include "../event.hpp"
#include "../grid.hpp"
#include "../grid_object.hpp"
#include "../stats_tracker.hpp"
#include "agent.hpp"
#include "constants.hpp"
#include "metta_object.hpp"

class FreezeTower : public MettaObject {
private:
  void maybe_start_attack() {
    // We can't start attacking if there's no event manager
    assert(this->event_manager != nullptr);
    // We also need to have an id to schedule the attack event
    assert(this->id != 0);
    if (this->cooling_down) {
      return;
    }

    // Look for agents within attack range
    Agent* target = find_target_in_range();
    if (target != nullptr) {
      // Attack the target
      attack_target(target);

      // Start cooldown
      if (this->cooldown > 0) {
        this->cooling_down = true;
        stats.incr("freeze_tower.cooldown.started");
        this->event_manager->schedule_event(EventType::FreezeTowerAttack, this->cooldown, this->id, 0);
      }
    }
  }

  Agent* find_target_in_range() {
    // Search for agents within attack range (similar to laser attack)
    for (int distance = 1; distance <= this->attack_range; distance++) {
      for (int offset = -1; offset <= 1; offset++) {
        // Check all 4 directions
        for (int orientation = 0; orientation < 4; orientation++) {
          GridLocation target_loc = this->event_manager->grid->relative_location(
              this->location, static_cast<Orientation>(orientation), distance, offset);

          target_loc.layer = GridLayer::Agent_Layer;
          Agent* agent_target = static_cast<Agent*>(this->event_manager->grid->object_at(target_loc));

          if (agent_target && agent_target->frozen == 0) {
            // Found an unfrozen agent in range
            return agent_target;
          }
        }
      }
    }
    return nullptr;
  }

  void attack_target(Agent* target) {
    // Freeze the target
    target->frozen = target->freeze_duration;

    // Track stats
    stats.incr("freeze_tower.attacks");
    stats.incr("freeze_tower.attacks." + target->group_name);
    target->stats.incr("freeze_tower.frozen_by");
    target->stats.incr("freeze_tower.frozen_by." + target->group_name);
  }

public:
  unsigned char attack_range;    // How far the tower can attack
  unsigned char cooldown;        // Time to wait between attacks
  bool cooling_down;             // Currently in cooldown phase
  unsigned char color;
  EventManager* event_manager;
  StatsTracker stats;

  FreezeTower(GridCoord r, GridCoord c, ObjectConfig cfg) {
    GridObject::init(ObjectType::FreezeTowerT, GridLocation(r, c, GridLayer::Object_Layer));
    MettaObject::init_mo(cfg);

    this->attack_range = cfg.count("attack_range") ? cfg["attack_range"] : 3;
    this->cooldown = cfg["cooldown"];
    this->color = cfg.count("color") ? cfg["color"] : 0;
    this->cooling_down = false;
    this->event_manager = nullptr;
  }

  void set_event_manager(EventManager* event_manager) {
    this->event_manager = event_manager;
    this->maybe_start_attack();
  }

  void finish_cooldown() {
    this->cooling_down = false;
    stats.incr("freeze_tower.cooldown.completed");
    this->maybe_start_attack();
  }

  // Called every step to check for new targets
  void step() {
    if (!this->cooling_down) {
      this->maybe_start_attack();
    }
  }

  virtual vector<PartialObservationToken> obs_features() const override {
    vector<PartialObservationToken> features;
    features.push_back({ObservationFeature::TypeId, _type_id});
    features.push_back({ObservationFeature::Hp, hp});
    features.push_back({ObservationFeature::Color, color});
    features.push_back({ObservationFeature::ConvertingOrCoolingDown, this->cooling_down});
    return features;
  }

  void obs(ObsType* obs, const std::vector<uint8_t>& offsets) const override {
    obs[offsets[0]] = 1;
    obs[offsets[1]] = _type_id;
    obs[offsets[2]] = this->hp;
    obs[offsets[3]] = this->color;
    obs[offsets[4]] = this->cooling_down;
    // Fill remaining inventory slots with 0 for consistency with converters
    for (unsigned int i = 0; i < InventoryItem::InventoryItemCount; i++) {
      obs[offsets[5 + i]] = 0;
    }
  }

  static std::vector<std::string> feature_names() {
    std::vector<std::string> names;
    names.push_back("freeze_tower");
    names.push_back("type_id");
    names.push_back("hp");
    names.push_back("color");
    names.push_back("cooling_down");
    // Add dummy inventory names for consistency
    for (unsigned int i = 0; i < InventoryItem::InventoryItemCount; i++) {
      names.push_back("inv:" + InventoryItemNames[i]);
    }
    return names;
  }
};

#endif  // METTAGRID_METTAGRID_OBJECTS_FREEZE_TOWER_HPP_
