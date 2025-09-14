#ifndef OBJECTS_GROUP_CONVERTER_HPP_
#define OBJECTS_GROUP_CONVERTER_HPP_

#include <algorithm>
#include <cassert>
#include <string>
#include <vector>
#include <map>
#include <set>

#include "../event.hpp"
#include "../stats_tracker.hpp"
#include "agent.hpp"
#include "constants.hpp"
#include "converter_config.hpp"
#include "has_inventory.hpp"
#include "../grid.hpp"

// GroupConverter extends the Converter functionality to require multiple agents
// positioned around it to activate. Used for the nano-assembler in Cogs vs Clips.
class GroupConverter : public HasInventory {
private:
  void maybe_start_converting(Grid* grid) {
    assert(this->event_manager);
    assert(this->id != 0);

    if (this->converting || this->cooling_down) {
      return;
    }

    // Check if the converter has reached max conversions
    if (this->max_conversions >= 0 && this->conversions_completed >= this->max_conversions) {
      stats.incr("conversions.permanent_stop");
      return;
    }

    // Check for positioned agents and determine recipe
    auto positioned_agents = get_positioned_agents(grid);
    if (positioned_agents.empty()) {
      stats.incr("blocked.no_agents");
      return;
    }

    uint8_t recipe_pattern = calculate_recipe_pattern(positioned_agents);
    auto recipe_it = recipes.find(recipe_pattern);
    if (recipe_it == recipes.end()) {
      stats.incr("blocked.unknown_recipe");
      return;
    }

    const auto& recipe = recipe_it->second;

    // Check if agents have required resources (pooled)
    std::map<InventoryItem, InventoryQuantity> available_resources;
    for (auto agent : positioned_agents) {
      for (const auto& [item, amount] : agent->inventory) {
        available_resources[item] += amount;
      }
    }

    // Check if we have enough pooled resources
    for (const auto& [item, required_amount] : recipe.input_resources) {
      if (available_resources[item] < required_amount) {
        stats.incr("blocked.insufficient_pooled_resources");
        return;
      }
    }

    // Consume resources from agents (starting from north, clockwise)
    std::map<InventoryItem, InventoryQuantity> remaining_to_consume = recipe.input_resources;
    std::vector<Agent*> sorted_agents = sort_agents_clockwise(positioned_agents);

    for (auto agent : sorted_agents) {
      for (auto& [item, remaining] : remaining_to_consume) {
        if (remaining > 0 && agent->inventory.count(item) > 0) {
          InventoryQuantity available = agent->inventory[item];
          InventoryQuantity to_take = std::min(available, remaining);
          agent->update_inventory(item, -to_take);
          remaining -= to_take;
          stats.add(stats.resource_name(item) + ".consumed_from_agent", to_take);
        }
      }
    }

    // Store the current recipe for output
    current_recipe = recipe;
    current_agents = sorted_agents;

    // Start converting
    this->converting = true;
    stats.incr("conversions.started");
    stats.add("recipe.pattern_" + std::to_string(recipe_pattern) + ".executed", 1);

    this->event_manager->schedule_event(EventType::FinishConverting, this->conversion_ticks, this->id, 0);
  }

  std::vector<Agent*> get_positioned_agents(Grid* grid) {
    std::vector<Agent*> agents;

    // Check all 8 positions around the group converter
    const std::vector<std::pair<int, int>> positions = {
      {-1, 0},  // North (0)
      {-1, 1},  // Northeast (1)
      {0, 1},   // East (2)
      {1, 1},   // Southeast (3)
      {1, 0},   // South (4)
      {1, -1},  // Southwest (5)
      {0, -1},  // West (6)
      {-1, -1}  // Northwest (7)
    };

    GridCoord base_row = this->location.r;
    GridCoord base_col = this->location.c;

    for (int i = 0; i < 8; i++) {
      GridCoord check_row = base_row + positions[i].first;
      GridCoord check_col = base_col + positions[i].second;

      if (check_row >= 0 && check_row < grid->height &&
          check_col >= 0 && check_col < grid->width) {
        GridLocation check_loc(check_row, check_col, GridLayer::AgentLayer);
        GridObject* obj = grid->object_at(check_loc);
        Agent* agent = dynamic_cast<Agent*>(obj);
        if (agent != nullptr) {
          agents.push_back(agent);
        }
      }
    }

    return agents;
  }

  uint8_t calculate_recipe_pattern(const std::vector<Agent*>& positioned_agents) {
    uint8_t pattern = 0;

    const std::vector<std::pair<int, int>> positions = {
      {-1, 0},  // North (bit 0)
      {-1, 1},  // Northeast (bit 1)
      {0, 1},   // East (bit 2)
      {1, 1},   // Southeast (bit 3)
      {1, 0},   // South (bit 4)
      {1, -1},  // Southwest (bit 5)
      {0, -1},  // West (bit 6)
      {-1, -1}  // Northwest (bit 7)
    };

    GridCoord base_row = this->location.r;
    GridCoord base_col = this->location.c;

    for (auto agent : positioned_agents) {
      GridCoord agent_row = agent->location.r;
      GridCoord agent_col = agent->location.c;

      int row_diff = agent_row - base_row;
      int col_diff = agent_col - base_col;

      for (int i = 0; i < 8; i++) {
        if (positions[i].first == row_diff && positions[i].second == col_diff) {
          pattern |= (1 << i);
          break;
        }
      }
    }

    return pattern;
  }

  std::vector<Agent*> sort_agents_clockwise(const std::vector<Agent*>& agents) {
    std::vector<std::pair<Agent*, int>> agent_positions;

    const std::vector<std::pair<int, int>> positions = {
      {-1, 0},  // North (0)
      {-1, 1},  // Northeast (1)
      {0, 1},   // East (2)
      {1, 1},   // Southeast (3)
      {1, 0},   // South (4)
      {1, -1},  // Southwest (5)
      {0, -1},  // West (6)
      {-1, -1}  // Northwest (7)
    };

    GridCoord base_row = this->location.r;
    GridCoord base_col = this->location.c;

    for (auto agent : agents) {
      GridCoord agent_row = agent->location.r;
      GridCoord agent_col = agent->location.c;

      int row_diff = agent_row - base_row;
      int col_diff = agent_col - base_col;

      for (int i = 0; i < 8; i++) {
        if (positions[i].first == row_diff && positions[i].second == col_diff) {
          agent_positions.push_back({agent, i});
          break;
        }
      }
    }

    // Sort by position index (clockwise from north)
    std::sort(agent_positions.begin(), agent_positions.end(),
              [](const auto& a, const auto& b) { return a.second < b.second; });

    std::vector<Agent*> sorted;
    for (const auto& [agent, _] : agent_positions) {
      sorted.push_back(agent);
    }

    return sorted;
  }

public:
  struct Recipe {
    std::map<InventoryItem, InventoryQuantity> input_resources;
    std::map<InventoryItem, InventoryQuantity> output_resources;

    // Default constructor
    Recipe() : input_resources(), output_resources() {}

    // Parameterized constructor
    Recipe(const std::map<InventoryItem, InventoryQuantity>& inputs,
           const std::map<InventoryItem, InventoryQuantity>& outputs)
      : input_resources(inputs), output_resources(outputs) {}
  };

  std::map<uint8_t, Recipe> recipes;  // Pattern -> Recipe mapping
  Recipe current_recipe;
  std::vector<Agent*> current_agents;

  short max_output;
  short max_conversions;
  unsigned short conversion_ticks;
  unsigned short cooldown;
  bool converting;
  bool cooling_down;
  unsigned char color;
  EventManager* event_manager;
  StatsTracker stats;
  unsigned short conversions_completed;

  GroupConverter(GridCoord r, GridCoord c, const ConverterConfig& cfg)
      : max_output(cfg.max_output),
        max_conversions(cfg.max_conversions),
        conversion_ticks(cfg.conversion_ticks),
        cooldown(cfg.cooldown),
        converting(false),
        cooling_down(false),
        color(cfg.color),
        event_manager(nullptr),
        current_recipe({}, {}),
        conversions_completed(0) {
    GridObject::init(cfg.type_id, cfg.type_name, GridLocation(r, c, GridLayer::ObjectLayer));
  }

  void add_recipe(uint8_t pattern, const Recipe& recipe) {
    recipes[pattern] = recipe;
  }

  void set_event_manager(EventManager* event_manager_ptr) {
    this->event_manager = event_manager_ptr;
  }

  void try_activate(Grid* grid) {
    this->maybe_start_converting(grid);
  }

  void finish_converting() {
    this->converting = false;
    stats.incr("conversions.completed");

    if (this->max_conversions >= 0) {
      this->conversions_completed++;
    }

    // Deliver output to the first agent (activating agent)
    if (!current_agents.empty()) {
      Agent* activating_agent = current_agents[0];
      for (const auto& [item, amount] : current_recipe.output_resources) {
        activating_agent->update_inventory(item, amount);
        stats.add(stats.resource_name(item) + ".produced", amount);
      }
    }

    if (this->cooldown > 0) {
      this->cooling_down = true;
      stats.incr("cooldown.started");
      this->event_manager->schedule_event(EventType::CoolDown, this->cooldown, this->id, 0);
    }
  }

  void finish_cooldown() {
    this->cooling_down = false;
    stats.incr("cooldown.completed");
  }

  std::vector<PartialObservationToken> obs_features() const override {
    std::vector<PartialObservationToken> features;

    features.reserve(3);
    features.push_back({ObservationFeature::TypeId, static_cast<ObservationType>(this->type_id)});
    features.push_back({ObservationFeature::Color, static_cast<ObservationType>(this->color)});
    features.push_back({ObservationFeature::ConvertingOrCoolingDown,
                        static_cast<ObservationType>(this->converting || this->cooling_down)});

    return features;
  }
};

#endif  // OBJECTS_GROUP_CONVERTER_HPP_