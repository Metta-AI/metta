#ifndef CONVERTER_HPP
#define CONVERTER_HPP

#include <cassert>
#include <cstdint>
#include <string>
#include <vector>

#include "constants.hpp"
#include "event_manager.hpp"
#include "grid_object.hpp"
#include "objects/agent.hpp"
#include "objects/has_inventory.hpp"
#include "objects/metta_object.hpp"
#include "types.hpp"

// Converter class definition
class Converter : public HasInventory {
private:
  // This should be called any time the converter could start converting
  void maybe_start_converting() {
    // We can't start converting if there's no event manager, since we won'tbe able to schedule the finishing event.
    assert(this->event_manager != nullptr);
    // We also need to have an id to schedule the finishing event. If our id id zero, we probably haven't been added to
    // the grid yet.
    assert(this->id != 0);
    if (this->converting || this->cooling_down) {
      return;
    }
    // Check if the converter is already at max output.
    uint16_t total_output = 0;
    for (uint32_t i = 0; i < InventoryItem::InventoryCount; i++) {
      if (this->recipe_output[i] > 0) {
        total_output += this->inventory[i];
      }
    }
    if (total_output >= this->max_output) {
      return;
    }
    // Check if the converter has enough input.
    for (uint32_t i = 0; i < InventoryItem::InventoryCount; i++) {
      if (this->inventory[i] < this->recipe_input[i]) {
        return;
      }
    }
    // produce.
    for (uint32_t i = 0; i < InventoryItem::InventoryCount; i++) {
      this->inventory[i] -= this->recipe_input[i];
    }
    // All the previous returns were "we don't start converting". This one is us starting to convert.
    this->converting = true;
    this->event_manager->schedule_event(Events::FinishConverting, this->conversion_ticks, this->id, 0);
  }

public:
  std::vector<uint8_t> recipe_input;
  std::vector<uint8_t> recipe_output;
  uint16_t max_output;
  uint8_t conversion_ticks;
  uint8_t cooldown;
  bool converting;
  bool cooling_down;
  uint8_t color;
  EventManager* event_manager;

  Converter(GridCoord r, GridCoord c, ObjectConfig cfg, TypeId type_id) {
    GridObject::init(type_id, GridLocation(r, c, GridLayer::Object_Layer));
    MettaObject::set_hp(cfg);
    HasInventory::init_has_inventory(cfg);
    this->recipe_input.resize(InventoryItem::InventoryCount);
    this->recipe_output.resize(InventoryItem::InventoryCount);
    for (uint32_t i = 0; i < InventoryItem::InventoryCount; i++) {
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
    uint8_t initial_items = cfg["initial_items"];
    for (uint32_t i = 0; i < InventoryItem::InventoryCount; i++) {
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

    // Add output to inventory
    for (uint32_t i = 0; i < InventoryItem::InventoryCount; i++) {
      if (this->recipe_output[i] > 0) {
        HasInventory::update_inventory(static_cast<InventoryItem>(i), this->recipe_output[i]);
      }
    }

    if (this->cooldown > 0) {
      // Start cooldown phase
      this->cooling_down = true;
      this->event_manager->schedule_event(Events::CoolDown, this->cooldown, this->id, 0);
    } else if (this->cooldown == 0) {
      // No cooldown, try to start converting again immediately
      this->maybe_start_converting();
    } else if (this->cooldown < 0) {
      // Negative cooldown means never convert again
      this->cooling_down = true;
    }
  }

  void finish_cooldown() {
    this->cooling_down = false;
    this->maybe_start_converting();
  }

  void update_inventory(InventoryItem item, int16_t amount) override {
    HasInventory::update_inventory(item, amount);
    this->maybe_start_converting();
  }

  virtual void obs(c_observations_type* obs) const override {
    HasInventory::obs(obs);

    // Map object type to corresponding feature
    GridFeature objectTypeFeature;
    switch (_type_id) {
      case ObjectType::AgentT:
        objectTypeFeature = GridFeature::AGENT_TYPE;
        break;
      case ObjectType::WallT:
        objectTypeFeature = GridFeature::WALL_TYPE;
        break;
      case ObjectType::MineT:
        objectTypeFeature = GridFeature::MINE_TYPE;
        break;
      case ObjectType::GeneratorT:
        objectTypeFeature = GridFeature::GENERATOR_TYPE;
        break;
      case ObjectType::AltarT:
        objectTypeFeature = GridFeature::ALTAR_TYPE;
        break;
      case ObjectType::ArmoryT:
        objectTypeFeature = GridFeature::ARMORY_TYPE;
        break;
      case ObjectType::LaseryT:
        objectTypeFeature = GridFeature::LASERY_TYPE;
        break;
      case ObjectType::LabT:
        objectTypeFeature = GridFeature::LAB_TYPE;
        break;
      case ObjectType::FactoryT:
        objectTypeFeature = GridFeature::FACTORY_TYPE;
        break;
      case ObjectType::TempleT:
        objectTypeFeature = GridFeature::TEMPLE_TYPE;
        break;
      case ObjectType::GenericConverterT:
        objectTypeFeature = GridFeature::CONVERTER_TYPE;
        break;
      default:
        objectTypeFeature = GridFeature::CONVERTER_TYPE;  // Default case
    }

    // Converter-specific features
    encode(obs, objectTypeFeature, 1);
    encode(obs, GridFeature::COLOR, this->color);
    encode(obs, GridFeature::CONVERTING, this->converting || this->cooling_down);

    // Map inventory items to their corresponding general inventory features
    const GridFeature invFeatures[] = {GridFeature::INV_ORE_RED,
                                       GridFeature::INV_ORE_BLUE,
                                       GridFeature::INV_ORE_GREEN,
                                       GridFeature::INV_BATTERY,
                                       GridFeature::INV_HEART,
                                       GridFeature::INV_ARMOR,
                                       GridFeature::INV_LASER,
                                       GridFeature::INV_BLUEPRINT};

    // Inventory features
    for (uint32_t i = 0; i < InventoryItem::InventoryCount; i++) {
      encode(obs, invFeatures[i], this->inventory[i]);
    }
  }
};

#endif  // CONVERTER_HPP