#ifndef PACKAGES_METTAGRID_CPP_INCLUDE_METTAGRID_OBJECTS_CONVERTER_HPP_
#define PACKAGES_METTAGRID_CPP_INCLUDE_METTAGRID_OBJECTS_CONVERTER_HPP_

#include <cassert>
#include <string>
#include <vector>

#include "core/event.hpp"
#include "objects/agent.hpp"
#include "objects/constants.hpp"
#include "objects/converter_config.hpp"
#include "objects/has_inventory.hpp"

class Converter : public GridObject, public HasInventory {
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
      return;
    }
    // Check if the converter is already at max output.
    unsigned short total_output = 0;
    for (const auto& [item, amount] : this->inventory.get()) {
      if (this->output_resources.count(item) > 0) {
        total_output += amount;
      }
    }
    if (this->max_output >= 0 && total_output >= this->max_output) {
      return;
    }
    // Check if the converter has enough input.
    for (const auto& [item, input_amount] : this->input_resources) {
      if (this->inventory.amount(item) < input_amount) {
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
      this->inventory.update(item, -amount);
    }
    // All the previous returns were "we don't start converting".
    // This one is us starting to convert.
    this->converting = true;
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
  ObservationType input_recipe_offset;
  ObservationType output_recipe_offset;
  unsigned short conversions_completed;

  Converter(GridCoord r, GridCoord c, const ConverterConfig& cfg)
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
        event_manager(nullptr),
        input_recipe_offset(cfg.input_recipe_offset),
        output_recipe_offset(cfg.output_recipe_offset),
        conversions_completed(0) {
    GridObject::init(cfg.type_id, cfg.type_name, GridLocation(r, c, GridLayer::ObjectLayer), cfg.tag_ids);

    // Initialize inventory with initial_resource_count for all output types
    for (const auto& [item, _] : this->output_resources) {
      this->inventory.update(item, cfg.initial_resource_count);
    }
  }

  void set_event_manager(EventManager* event_manager_ptr) {
    this->event_manager = event_manager_ptr;
    this->maybe_start_converting();
  }

  void finish_converting() {
    this->converting = false;

    // Only increment the counter when tracking conversion limits
    if (this->max_conversions >= 0) {
      this->conversions_completed++;
    }

    // Add output to inventory
    for (const auto& [item, amount] : this->output_resources) {
      this->inventory.update(item, amount);
    }

    if (this->cooldown > 0) {
      // Start cooldown phase
      this->cooling_down = true;
      this->event_manager->schedule_event(EventType::CoolDown, this->cooldown, this->id, 0);
    } else if (this->cooldown == 0) {
      // No cooldown, try to start converting again immediately
      this->maybe_start_converting();
    }
  }

  void finish_cooldown() {
    this->cooling_down = false;
    this->maybe_start_converting();
  }

  InventoryDelta update_inventory(InventoryItem item, InventoryDelta attempted_delta) override {
    InventoryDelta delta = this->inventory.update(item, attempted_delta);
    this->maybe_start_converting();
    return delta;
  }

  std::vector<PartialObservationToken> obs_features() const override {
    std::vector<PartialObservationToken> features;

    // Calculate the capacity needed
    // We push 3 fixed features + inventory items + (optionally) recipe inputs and outputs + tags
    size_t capacity = 3 + this->inventory.get().size() + this->tag_ids.size();
    if (this->recipe_details_obs) {
      capacity += this->input_resources.size() + this->output_resources.size();
    }
    features.reserve(capacity);

    features.push_back({ObservationFeature::TypeId, static_cast<ObservationType>(this->type_id)});
    features.push_back({ObservationFeature::Color, static_cast<ObservationType>(this->color)});
    features.push_back({ObservationFeature::ConvertingOrCoolingDown,
                        static_cast<ObservationType>(this->converting || this->cooling_down)});

    // Add current inventory (inv:resource)
    for (const auto& [item, amount] : this->inventory.get()) {
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

    // Emit tag features
    for (int tag_id : tag_ids) {
      features.push_back({ObservationFeature::Tag, static_cast<ObservationType>(tag_id)});
    }

    return features;
  }
};

#endif  // PACKAGES_METTAGRID_CPP_INCLUDE_METTAGRID_OBJECTS_CONVERTER_HPP_
