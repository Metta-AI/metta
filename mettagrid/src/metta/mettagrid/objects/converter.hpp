#ifndef OBJECTS_CONVERTER_HPP_
#define OBJECTS_CONVERTER_HPP_

#include <cassert>
#include <string>
#include <vector>

#include "../event.hpp"
#include "../grid_object.hpp"
#include "../stats_tracker.hpp"
#include "agent.hpp"
#include "constants.hpp"
#include "has_inventory.hpp"

// Forward declaration
class MettaGrid;

// #MettagridConfig
struct ConverterConfig : public GridObjectConfig {
  ConverterConfig(TypeId type_id,
                  const std::string& type_name,
                  const std::map<InventoryItem, InventoryQuantity>& input_resources,
                  const std::map<InventoryItem, InventoryQuantity>& output_resources,
                  short max_output,
                  unsigned short conversion_ticks,
                  unsigned short cooldown,
                  InventoryQuantity initial_resource_count,
                  ObservationType color,
                  bool cyclical = false,
                  unsigned short phase = 0,
                  bool recipe_details_obs = false,
                  ObservationType input_recipe_offset = 0,
                  ObservationType output_recipe_offset = 0)
      : GridObjectConfig(type_id, type_name),
        input_resources(input_resources),
        output_resources(output_resources),
        max_output(max_output),
        conversion_ticks(conversion_ticks),
        cooldown(cooldown),
        initial_resource_count(initial_resource_count),
        color(color),
        cyclical(cyclical),
        phase(phase),
        recipe_details_obs(recipe_details_obs),
        input_recipe_offset(input_recipe_offset),
        output_recipe_offset(output_recipe_offset) {}

  std::map<InventoryItem, InventoryQuantity> input_resources;
  std::map<InventoryItem, InventoryQuantity> output_resources;
  short max_output;
  unsigned short conversion_ticks;
  unsigned short cooldown;
  InventoryQuantity initial_resource_count;
  ObservationType color;
  bool cyclical;
  unsigned short phase;
  bool recipe_details_obs;
  ObservationType input_recipe_offset;
  ObservationType output_recipe_offset;
};

class Converter : public HasInventory {
private:
  // This should be called any time the converter could start converting. E.g.,
  // when things are added to its input, and when it finishes converting.
  void maybe_start_converting() {
    // We can't start converting if there's no event manager, since we won't
    // be able to schedule the finishing event.
    assert(this->event_manager);
    // We also need to have an id to schedule the finishing event. If our id
    // is zero, we probably haven't been added to the grid yet.
    assert(this->id != 0);
    if (this->converting || this->cooling_down) {
      return;
    }
    // Check if the converter is already at max output.
    unsigned short total_output = 0;
    for (const auto& [item, amount] : this->inventory) {
      if (this->output_resources.count(item) > 0) {
        total_output += amount;
      }
    }
    if (this->max_output >= 0 && total_output >= this->max_output) {
      stats.incr("blocked.output_full");
      return;
    }
    // Check if the converter has enough input.
    for (const auto& [item, input_amount] : this->input_resources) {
      if (this->inventory.count(item) == 0 || this->inventory.at(item) < input_amount) {
        stats.incr("blocked.insufficient_input");
        return;
      }
    }
    // produce.
    // Get the amounts to consume from input, so we don't update the inventory
    // while iterating over it.
    std::map<InventoryItem, uint8_t> amounts_to_consume;
    for (const auto& [item, input_amount] : this->input_resources) {
      amounts_to_consume[item] = input_amount;
    }

    for (const auto& [item, amount] : amounts_to_consume) {
      // Don't call update_inventory here, because it will call maybe_start_converting again,
      // which will cause an infinite loop.
      this->inventory[item] -= amount;
      if (this->inventory[item] == 0) {
        this->inventory.erase(item);
      }
      stats.add(stats.inventory_item_name(item) + ".consumed", static_cast<float>(amount));
    }
    // All the previous returns were "we don't start converting".
    // This one is us starting to convert.
    this->converting = true;
    stats.incr("conversions.started");

        this->event_manager->schedule_event(EventType::FinishConverting, this->conversion_ticks, this->id, 0);
  }

public:
  std::map<InventoryItem, InventoryQuantity> input_resources;
  std::map<InventoryItem, InventoryQuantity> output_resources;
  // The converter won't convert if its output already has this many things of
  // the type it produces. This may be clunky in some cases, but the main usage
  // is to make Mines (etc) have a maximum output.
  // -1 means no limit
  short max_output;
  unsigned short conversion_ticks;  // Time to produce output
  unsigned short cooldown;          // Time to wait after producing before starting again

      // Pack flags into bitfields for memory efficiency
  unsigned char converting : 1;         // Currently in production phase
  unsigned char cooling_down : 1;       // Currently in cooldown phase
  unsigned char cyclical : 1;           // Whether to auto-empty inventory after cooldown
  unsigned char has_phase : 1;          // Whether this converter has a phase delay
  unsigned char _reserved : 4;          // Reserved for future flags

    unsigned short phase;                  // Initial delay before first conversion
  unsigned char color;
  bool recipe_details_obs;
  EventManager* event_manager;
  StatsTracker stats;
  ObservationType input_recipe_offset;
  ObservationType output_recipe_offset;

  Converter(GridCoord r, GridCoord c, const ConverterConfig& cfg)
      : input_resources(cfg.input_resources),
        output_resources(cfg.output_resources),
        max_output(cfg.max_output),
        conversion_ticks(cfg.conversion_ticks),
        cooldown(cfg.cooldown),
        color(cfg.color),
        recipe_details_obs(cfg.recipe_details_obs),
        event_manager(nullptr),
        converting(false),
        cooling_down(false),
        cyclical(cfg.cyclical),
        phase(cfg.phase),
        input_recipe_offset(cfg.input_recipe_offset),
        output_recipe_offset(cfg.output_recipe_offset) {
    GridObject::init(cfg.type_id, cfg.type_name, GridLocation(r, c, GridLayer::ObjectLayer));

    // Initialize inventory with initial_resource_count for all output types
    for (const auto& [item, _] : this->output_resources) {
      HasInventory::update_inventory(item, static_cast<InventoryDelta>(cfg.initial_resource_count));
    }
  }

  void set_event_manager(EventManager* event_manager_ptr) {
    this->event_manager = event_manager_ptr;

    // If there's a phase delay, schedule the initial start
    if (this->phase > 0) {
      // Schedule a "start phase" by scheduling a fake cooldown completion
      this->cooling_down = true;  // Prevent immediate conversion
      this->event_manager->schedule_event(EventType::CoolDown, this->phase, this->id, 0);
    } else {
      // No phase delay, start immediately
      this->maybe_start_converting();
    }
  }



  void finish_converting() {
    this->converting = false;
    stats.incr("conversions.completed");

    // Add output to inventory
    for (const auto& [item, amount] : this->output_resources) {
      HasInventory::update_inventory(item, static_cast<InventoryDelta>(amount));
      stats.add(stats.inventory_item_name(item) + ".produced", static_cast<float>(amount));
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
    // Cyclical converters auto-empty on cooldown completion
    if (this->cyclical) {
      // For single-output converters (common case), optimize the clear
      if (this->output_resources.size() == 1) {
        auto output_item = this->output_resources.begin()->first;
        auto it = this->inventory.find(output_item);
        if (it != this->inventory.end()) {
          this->inventory.erase(it);
          stats.incr("inventory.auto_emptied");
        }
      } else {
        // Multi-output case: clear all output items
        bool emptied_something = false;
        for (const auto& [item, _] : this->output_resources) {
          auto it = this->inventory.find(item);
          if (it != this->inventory.end()) {
            this->inventory.erase(it);
            emptied_something = true;
          }
        }
        if (emptied_something) {
          stats.incr("inventory.auto_emptied");
        }
      }
    }

    this->cooling_down = false;
    stats.incr("cooldown.completed");
    this->maybe_start_converting();
  }



  InventoryDelta update_inventory(InventoryItem item, InventoryDelta attempted_delta) override {
    InventoryDelta delta = HasInventory::update_inventory(item, attempted_delta);
    if (delta != 0) {
      if (delta > 0) {
        stats.add(stats.inventory_item_name(item) + ".added", delta);
      } else {
        stats.add(stats.inventory_item_name(item) + ".removed", -delta);
      }
    }
    this->maybe_start_converting();
    return delta;
  }

  std::vector<PartialObservationToken> obs_features() const override {
    std::vector<PartialObservationToken> features;

    // Calculate the capacity needed
    // We push 3 fixed features + inventory items + (optionally) recipe inputs and outputs
    size_t capacity = 3 + this->inventory.size();
    if (this->recipe_details_obs) {
      capacity += this->input_resources.size() + this->output_resources.size();
    }
    features.reserve(capacity);

    features.push_back({ObservationFeature::TypeId, static_cast<ObservationType>(this->type_id)});
    features.push_back({ObservationFeature::Color, static_cast<ObservationType>(this->color)});
    features.push_back({ObservationFeature::ConvertingOrCoolingDown,
                        static_cast<ObservationType>(this->converting || this->cooling_down)});

    // Add current inventory (inv:resource)
    for (const auto& [item, amount] : this->inventory) {
      // inventory should only contain non-zero amounts
      assert(amount > 0);
      features.push_back(
          {static_cast<ObservationType>(item + InventoryFeatureOffset), static_cast<ObservationType>(amount)});
    }

    // Add recipe details if configured to do so
    if (this->recipe_details_obs) {
      // Add recipe inputs (input:resource) - only non-zero values
      for (const auto& [item, amount] : this->input_resources) {
        if (amount > 0) {
          features.push_back({static_cast<ObservationType>(input_recipe_offset + item), static_cast<ObservationType>(amount)});
        }
      }

      // Add recipe outputs (output:resource) - only non-zero values
      for (const auto& [item, amount] : this->output_resources) {
        if (amount > 0) {
          features.push_back({static_cast<ObservationType>(output_recipe_offset + item), static_cast<ObservationType>(amount)});
        }
      }
    }

    return features;
  }
};

#endif  // OBJECTS_CONVERTER_HPP_
