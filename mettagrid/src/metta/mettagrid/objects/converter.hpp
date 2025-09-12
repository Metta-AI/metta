#ifndef OBJECTS_CONVERTER_HPP_
#define OBJECTS_CONVERTER_HPP_

#include <cassert>
#include <climits>
#include <random>
#include <string>
#include <vector>

#include "../event.hpp"
#include "../stats_tracker.hpp"
#include "agent.hpp"
#include "constants.hpp"
#include "converter_config.hpp"
#include "has_inventory.hpp"
#include "inventory_list.hpp"

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
    // Check if the converter has reached max conversions
    if (this->max_conversions >= 0 && this->conversions_completed >= this->max_conversions) {
      stats.incr("conversions.permanent_stop");
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
      stats.add(stats.resource_name(item) + ".consumed", amount);
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
  short max_conversions;
  unsigned short conversion_ticks;  // Time to produce output
  unsigned short cooldown;          // Time to wait after producing before starting again
  bool converting;                  // Currently in production phase
  bool cooling_down;                // Currently in cooldown phase
  unsigned char color;
  bool recipe_details_obs;
  EventManager* event_manager;
  StatsTracker stats;
  ObservationType input_recipe_offset;
  ObservationType output_recipe_offset;
  unsigned short conversions_completed;

  // Inventory management with resource tracking
  InventoryList inventory_list;
  std::map<InventoryItem, InventoryQuantity> initial_inventory_config;

public:
  // Expose inventory and resource_instances for backward compatibility
  std::map<InventoryItem, InventoryQuantity>& inventory = inventory_list.inventory;
  std::map<uint64_t, ResourceInstance>& resource_instances = inventory_list.resource_instances;

  Converter(GridCoord r, GridCoord c, const ConverterConfig& cfg, EventManager* event_manager_ptr = nullptr, std::mt19937* rng_ptr = nullptr)
      : input_resources(cfg.input_resources),
        output_resources(cfg.output_resources),
        max_output(cfg.max_output),
        max_conversions(cfg.max_conversions),
        conversion_ticks(cfg.conversion_ticks),
        cooldown(cfg.cooldown),
        converting(false),
        cooling_down(false),
        color(cfg.color),
        recipe_details_obs(cfg.recipe_details_obs),
        event_manager(event_manager_ptr),
        input_recipe_offset(cfg.input_recipe_offset),
        output_recipe_offset(cfg.output_recipe_offset),
        conversions_completed(0),
        inventory_list(cfg.resource_loss_prob.empty() ? InventoryList() : InventoryList(event_manager_ptr, rng_ptr, cfg.resource_loss_prob)) {
    GridObject::init(cfg.type_id, cfg.type_name, GridLocation(r, c, GridLayer::ObjectLayer));

    // Store the initial inventory config for later initialization
    this->initial_inventory_config = std::map<InventoryItem, InventoryQuantity>();
    for (const auto& [item, _] : this->output_resources) {
      if (cfg.initial_resource_count > 0) {
        this->initial_inventory_config[item] = cfg.initial_resource_count;
      }
    }
  }

  void init() {
    // Initialize inventory and schedule resource loss events now that we have the correct ID
    this->inventory_list.populate_initial_inventory(this->initial_inventory_config, this->id);
  }

  // Implement HasInventory interface
  InventoryList& get_inventory_list() override {
    return inventory_list;
  }

  const InventoryList& get_inventory_list() const override {
    return inventory_list;
  }

  // Inventory access method (required by some action handlers)
  bool inventory_is_accessible() const override {
    return true;  // Converters always have accessible inventory
  }


  void finish_converting() {
    this->converting = false;
    // Increment the stat unconditionally
    stats.incr("conversions.completed");

    // Only increment the counter when tracking conversion limits
    if (this->max_conversions >= 0) {
      this->conversions_completed++;
    }

    // Add output to inventory
    for (const auto& [item, amount] : this->output_resources) {
      update_inventory(item, amount);
      stats.add(stats.resource_name(item) + ".produced", amount);
    }

    if (this->cooldown > 0) {
      // Start cooldown phase
      this->cooling_down = true;
      stats.incr("cooldown.started");
      this->event_manager->schedule_event(EventType::CoolDown, this->cooldown, this->id, 0);
    } else if (this->cooldown == 0) {
      // No cooldown, try to start converting again immediately
      this->maybe_start_converting();
    }
  }

  void finish_cooldown() {
    this->cooling_down = false;
    stats.incr("cooldown.completed");
    this->maybe_start_converting();
  }

  InventoryDelta update_inventory(InventoryItem item, InventoryDelta delta) override {
    // Get the initial amount (0 if item doesn't exist)
    InventoryQuantity initial_amount = 0;
    auto inv_it = this->inventory.find(item);
    if (inv_it != this->inventory.end()) {
      initial_amount = inv_it->second;
    }

    // Calculate the new amount with clamping
    int new_amount_int = static_cast<int>(initial_amount + delta);
    InventoryQuantity new_amount = static_cast<InventoryQuantity>(std::clamp(
        new_amount_int, 0, static_cast<int>(std::numeric_limits<InventoryQuantity>::max())));

    InventoryDelta actual_delta = new_amount - initial_amount;

    // Handle inventory changes using InventoryList
    if (actual_delta != 0) {
      // Use InventoryList to handle the update with stochastic resource loss
      InventoryDelta inventory_delta = this->inventory_list.update_inventory(item, actual_delta, this->id);

      // Update stats
      if (inventory_delta > 0) {
        stats.add(stats.resource_name(item) + ".added", inventory_delta);
      } else if (inventory_delta < 0) {
        stats.add(stats.resource_name(item) + ".removed", -inventory_delta);
      }

      this->maybe_start_converting();
      return inventory_delta;
    }

    this->maybe_start_converting();
    return actual_delta;
  }

  std::vector<PartialObservationToken> obs_features() const {
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
          features.push_back(
              {static_cast<ObservationType>(input_recipe_offset + item), static_cast<ObservationType>(amount)});
        }
      }

      // Add recipe outputs (output:resource) - only non-zero values
      for (const auto& [item, amount] : this->output_resources) {
        if (amount > 0) {
          features.push_back(
              {static_cast<ObservationType>(output_recipe_offset + item), static_cast<ObservationType>(amount)});
        }
      }
    }

    return features;
  }
};

#endif  // OBJECTS_CONVERTER_HPP_
