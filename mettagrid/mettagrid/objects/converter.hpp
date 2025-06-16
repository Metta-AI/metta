#ifndef METTAGRID_METTAGRID_OBJECTS_CONVERTER_HPP_
#define METTAGRID_METTAGRID_OBJECTS_CONVERTER_HPP_

#include <cassert>
#include <string>
#include <vector>

#include "../event.hpp"
#include "../grid_object.hpp"
#include "../stats_tracker.hpp"
#include "agent.hpp"
#include "constants.hpp"
#include "has_inventory.hpp"
#include "metta_object.hpp"

class Converter : public HasInventory {
private:
  // This should be called any time the converter could start converting. E.g.,
  // when things are added to its input, and when it finishes converting.
  void maybe_start_converting() {
    // We can't start converting if there's no event manager, since we won't
    // be able to schedule the finishing event.
    assert(this->event_manager != nullptr);
    // We also need to have an id to schedule the finishing event. If our id
    // is zero, we probably haven't been added to the grid yet.
    assert(this->id != 0);
    if (this->converting || this->cooling_down) {
      return;
    }
    // Check if the converter is already at max output.
    unsigned short total_output = 0;
    for (unsigned int i = 0; i < InventoryItem::InventoryItemCount; i++) {
      if (this->recipe_output[i] > 0) {
        total_output += this->inventory[i];
      }
    }
    if (total_output >= this->max_output) {
      stats.incr("blocked.output_full");
      return;
    }
    // Check if the converter has enough input.
    for (unsigned int i = 0; i < InventoryItem::InventoryItemCount; i++) {
      if (this->inventory[i] < this->recipe_input[i]) {
        stats.incr("blocked.insufficient_input");
        return;
      }
    }
    // produce.
    for (unsigned int i = 0; i < InventoryItem::InventoryItemCount; i++) {
      this->inventory[i] -= this->recipe_input[i];
      if (this->recipe_input[i] > 0) {
        stats.add(InventoryItemNames[i] + ".consumed", this->recipe_input[i]);
      }
    }
    // All the previous returns were "we don't start converting".
    // This one is us starting to convert.
    this->converting = true;
    stats.incr("conversions.started");
    this->event_manager->schedule_event(EventType::FinishConverting, this->conversion_ticks, this->id, 0);
  }

public:
  vector<unsigned char> recipe_input;
  vector<unsigned char> recipe_output;
  // The converter won't convert if its output already has this many things of
  // the type it produces. This may be clunky in some cases, but the main usage
  // is to make Mines (etc) have a maximum output.
  unsigned short max_output;
  unsigned char conversion_ticks;  // Time to produce output
  unsigned char cooldown;          // Time to wait after producing before starting again
  bool converting;                 // Currently in production phase
  bool cooling_down;               // Currently in cooldown phase
  unsigned char color;
  EventManager* event_manager;
  StatsTracker stats;

  Converter(GridCoord r, GridCoord c, ObjectConfig cfg, TypeId type_id) {
    GridObject::init(type_id, GridLocation(r, c, GridLayer::Object_Layer));
    MettaObject::init_mo(cfg);
    HasInventory::init_has_inventory(cfg);
    this->recipe_input.resize(InventoryItem::InventoryItemCount);
    this->recipe_output.resize(InventoryItem::InventoryItemCount);
    for (unsigned int i = 0; i < InventoryItem::InventoryItemCount; i++) {
      this->recipe_input[i] = cfg["input_" + InventoryItemNames[i]];
      this->recipe_output[i] = cfg["output_" + InventoryItemNames[i]];
    }
    this->max_output = cfg["max_output"];
    this->conversion_ticks = cfg["conversion_ticks"];
    this->cooldown = cfg["cooldown"];
    this->color = cfg.count("color") ? cfg["color"] : 0;
    this->converting = false;
    this->cooling_down = false;

    // Initialize inventory with initial_items for all output types
    // Default to recipe_output values if initial_items is not present
    unsigned char initial_items = cfg["initial_items"];
    for (unsigned int i = 0; i < InventoryItem::InventoryItemCount; i++) {
      if (this->recipe_output[i] > 0) {
        HasInventory::update_inventory(static_cast<InventoryItem>(i), initial_items);
      }
    }
  }

  Converter(GridCoord r, GridCoord c, ObjectConfig cfg) : Converter(r, c, cfg, ObjectType::GenericConverterT) {}

  void set_event_manager(EventManager* event_manager) {
    this->event_manager = event_manager;
    this->maybe_start_converting();
  }

  void finish_converting() {
    this->converting = false;
    stats.incr("conversions.completed");

    // Add output to inventory
    for (unsigned int i = 0; i < InventoryItem::InventoryItemCount; i++) {
      if (this->recipe_output[i] > 0) {
        HasInventory::update_inventory(static_cast<InventoryItem>(i), this->recipe_output[i]);
        stats.add(InventoryItemNames[i] + ".produced", this->recipe_output[i]);
      }
    }

    if (this->cooldown > 0) {
      // Start cooldown phase
      this->cooling_down = true;
      stats.incr("cooldown.started");
      this->event_manager->schedule_event(EventType::CoolDown, this->cooldown, this->id, 0);
    } else if (this->cooldown == 0) {
      // No cooldown, try to start converting again immediately
      this->maybe_start_converting();
    } else if (this->cooldown < 0) {
      // Negative cooldown means never convert again
      this->cooling_down = true;
      stats.incr("conversions.permanent_stop");
    }
  }

  void finish_cooldown() {
    this->cooling_down = false;
    stats.incr("cooldown.completed");
    this->maybe_start_converting();
  }

  int update_inventory(InventoryItem item, short amount) override {
    int delta = HasInventory::update_inventory(item, amount);
    if (delta != 0) {
      if (delta > 0) {
        stats.add(InventoryItemNames[item] + ".added", delta);
      } else {
        stats.add(InventoryItemNames[item] + ".removed", -delta);
      }
    }
    this->maybe_start_converting();
    return delta;
  }

  virtual vector<PartialObservationToken> obs_features() const override {
    vector<PartialObservationToken> features;
    features.reserve(5 + InventoryItem::InventoryItemCount);
    features.push_back({ObservationFeature::TypeId, _type_id});
    features.push_back({ObservationFeature::Color, color});
    features.push_back({ObservationFeature::ConvertingOrCoolingDown, this->converting || this->cooling_down});
    for (uint8_t i = 0; i < InventoryItem::InventoryItemCount; i++) {
      if (inventory[i] > 0) {
        features.push_back({static_cast<uint8_t>(InventoryFeatureOffset + i), inventory[i]});
      }
    }
    return features;
  }

  void obs(ObsType* obs) const override {
    const auto offsets = Converter::offsets();
    size_t offset_idx = 0;
    obs[offsets[offset_idx++]] = _type_id;
    obs[offsets[offset_idx++]] = this->hp;
    obs[offsets[offset_idx++]] = this->color;
    obs[offsets[offset_idx++]] = this->converting || this->cooling_down;
    for (unsigned int i = 0; i < InventoryItem::InventoryItemCount; i++) {
      obs[offsets[offset_idx++]] = this->inventory[i];
    }
  }

  static std::vector<uint8_t> offsets() {
    std::vector<uint8_t> ids;
    // We use the same feature names for all converters, since this compresses
    // the observation space. At the moment we don't expose the recipe, since
    // we expect converters to be hard coded.
    ids.push_back(ObservationFeature::TypeId);
    ids.push_back(ObservationFeature::Hp);
    ids.push_back(ObservationFeature::Color);
    ids.push_back(ObservationFeature::ConvertingOrCoolingDown);
    for (unsigned int i = 0; i < InventoryItem::InventoryItemCount; i++) {
      ids.push_back(static_cast<uint8_t>(InventoryFeatureOffset + i));
    }
    return ids;
  }
};

#endif  // METTAGRID_METTAGRID_OBJECTS_CONVERTER_HPP_
