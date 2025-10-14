#ifndef PACKAGES_METTAGRID_CPP_INCLUDE_METTAGRID_OBJECTS_CONVERTER_HPP_
#define PACKAGES_METTAGRID_CPP_INCLUDE_METTAGRID_OBJECTS_CONVERTER_HPP_

#include <cassert>
#include <cmath>
#include <random>
#include <stdexcept>
#include <string>
#include <unordered_map>
#include <vector>

#include "core/event.hpp"
#include "core/probability.hpp"
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
    // Check if all output types are already at the per-type max_output.
    if (this->max_output >= 0) {
      bool all_outputs_capped = true;
      for (const auto& [item, _] : this->output_resources) {
        unsigned int have = this->inventory.amount(item);
        if (have < static_cast<unsigned int>(this->max_output)) {
          all_outputs_capped = false;
          break;
        }
      }
      if (all_outputs_capped) {
        return;
      }
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
    std::unordered_map<InventoryItem, uint8_t> amounts_to_consume;
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
  std::unordered_map<InventoryItem, InventoryQuantity> input_resources;
  std::unordered_map<InventoryItem, InventoryProbability> output_resources;
  // The converter won't convert if all of its output types already have this
  // many things (per-type cap). This may be clunky in some cases, but the main
  // usage is to make Mines (etc) have a maximum output per produced resource.
  // -1 means no limit
  short max_output;
  short max_conversions;
  unsigned short conversion_ticks;  // Time to produce output
  unsigned short cooldown;          // Time to wait after producing before starting again
  bool converting;                  // Currently in production phase
  bool cooling_down;                // Currently in cooldown phase
  bool recipe_details_obs;
  EventManager* event_manager;
  ObservationType input_recipe_offset;
  ObservationType output_recipe_offset;
  ObservationType output_recipe_fraction_offset;
  unsigned short conversions_completed;
  std::mt19937& rng;

  Converter(GridCoord r, GridCoord c, const ConverterConfig& cfg, std::mt19937& rng_engine)
      : GridObject(),
        HasInventory(InventoryConfig()),  // Converts have nothing to configure in their inventory. Yet.
        input_resources(cfg.input_resources),
        output_resources(cfg.output_resources),
        max_output(cfg.max_output),
        max_conversions(cfg.max_conversions),
        conversion_ticks(cfg.conversion_ticks),
        cooldown(cfg.cooldown),
        converting(false),
        cooling_down(false),
        recipe_details_obs(cfg.recipe_details_obs),
        event_manager(nullptr),
        input_recipe_offset(cfg.input_recipe_offset),
        output_recipe_offset(cfg.output_recipe_offset),
        output_recipe_fraction_offset(cfg.output_recipe_fraction_offset),
        conversions_completed(0),
        rng(rng_engine) {
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

    for (const auto& [item, amount] : this->output_resources) {
      if (amount <= 0.0f) {
        continue;
      }
      InventoryDelta delta = probabilistic_delta(amount, this->rng);
      if (delta <= 0) {
        continue;
      }
      // Clamp delta to per-type remaining capacity if a cap is enforced
      if (this->max_output >= 0) {
        unsigned int have = this->inventory.amount(item);
        int type_remaining = static_cast<int>(this->max_output) - static_cast<int>(have);
        if (type_remaining <= 0) {
          continue;
        }
        if (delta > type_remaining) {
          delta = static_cast<InventoryDelta>(type_remaining);
        }
      }
      if (delta > 0) {
        this->inventory.update(item, delta);
      }
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
    // We push 2 fixed features + inventory items + (optionally) recipe inputs and outputs + tags
    size_t capacity = 2 + this->inventory.get().size() + this->tag_ids.size();
    if (this->recipe_details_obs) {
      capacity += this->input_resources.size();
      capacity += this->output_resources.size();
      // Fractional outputs are encoded in a separate pass that can emit up to one token per resource.
      capacity += this->output_resources.size();
    }
    features.reserve(capacity);

    features.push_back({ObservationFeature::TypeId, static_cast<ObservationType>(this->type_id)});
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

      // Add recipe outputs (output:resource) - integral part only
      for (const auto& [item, amount] : this->output_resources) {
        if (amount >= 1.0f) {
          unsigned int floored_amount = static_cast<unsigned int>(amount);
          if (floored_amount > 0) {
            ObservationType encoded_amount =
                static_cast<ObservationType>(std::min(floored_amount, 255u));
            ObservationType feature_id =
                static_cast<ObservationType>(output_recipe_offset + item);
            features.push_back({feature_id, encoded_amount});
          }
        }
      }

      // Add recipe fractional outputs (output_frac:resource) for fractional remainders
      for (const auto& [item, amount] : this->output_resources) {
        if (amount <= 0.0f) {
          continue;
        }
        float fractional_part = amount - static_cast<float>(std::floor(amount));
        if (fractional_part <= 0.0f) {
          continue;
        }
        float scaled = std::ceil(fractional_part * 255.0f);
        unsigned int bucket = static_cast<unsigned int>(std::min(std::max(scaled, 1.0f), 255.0f));
        ObservationType encoded_amount = static_cast<ObservationType>(bucket);
        ObservationType feature_id =
            static_cast<ObservationType>(output_recipe_fraction_offset + item);
        features.push_back({feature_id, encoded_amount});
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
