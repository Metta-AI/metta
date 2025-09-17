#ifndef OBJECTS_NANO_ASSEMBLER_HPP_
#define OBJECTS_NANO_ASSEMBLER_HPP_

#include <map>
#include <stdexcept>
#include <string>
#include <vector>

#include "../event.hpp"
#include "../grid.hpp"
#include "../stats_tracker.hpp"
#include "agent.hpp"
#include "constants.hpp"
#include "nano_assembler_config.hpp"
#include "recipe.hpp"
#include "types.hpp"
#include "usable.hpp"

// Forward declaration
class Agent;

class NanoAssembler : public Usable {
private:
  // Surrounding positions in deterministic order: NW, N, NE, W, E, SW, S, SE
  std::vector<std::pair<GridCoord, GridCoord>> get_surrounding_positions() const {
    GridCoord r = location.r;
    GridCoord c = location.c;
    return {
        {static_cast<GridCoord>(r - 1), static_cast<GridCoord>(c - 1)},
        {static_cast<GridCoord>(r - 1), c},
        {static_cast<GridCoord>(r - 1), static_cast<GridCoord>(c + 1)},
        {r, static_cast<GridCoord>(c - 1)},
        {r, static_cast<GridCoord>(c + 1)},
        {static_cast<GridCoord>(r + 1), static_cast<GridCoord>(c - 1)},
        {static_cast<GridCoord>(r + 1), c},
        {static_cast<GridCoord>(r + 1), static_cast<GridCoord>(c + 1)},
    };
  }

  // Helper function to convert surrounding agent positions to byte value
  // Returns a byte where each bit represents whether an agent is present
  // in the corresponding position around the assembler
  // Bit positions: 0=NW, 1=N, 2=NE, 3=W, 4=E, 5=SW, 6=S, 7=SE
  uint8_t get_agent_pattern_byte() const {
    if (!grid) return 0;

    uint8_t pattern = 0;
    std::vector<std::pair<GridCoord, GridCoord>> positions = get_surrounding_positions();

    for (size_t i = 0; i < positions.size(); i++) {
      GridCoord check_r = positions[i].first;
      GridCoord check_c = positions[i].second;

      if (check_r < grid->height && check_c < grid->width) {
        GridObject* obj = grid->object_at(GridLocation(check_r, check_c, GridLayer::AgentLayer));
        if (obj && dynamic_cast<Agent*>(obj)) {
          pattern |= static_cast<uint8_t>(1u << i);
        }
      }
    }

    return pattern;
  }

  // Get surrounding agents in a deterministic order (clockwise from NW)
  std::vector<Agent*> get_surrounding_agents() const {
    std::vector<Agent*> agents;
    if (!grid) return agents;

    std::vector<std::pair<GridCoord, GridCoord>> positions = get_surrounding_positions();

    for (const auto& pos : positions) {
      GridCoord check_r = pos.first;
      GridCoord check_c = pos.second;
      if (check_r < grid->height && check_c < grid->width) {
        GridObject* obj = grid->object_at(GridLocation(check_r, check_c, GridLayer::AgentLayer));
        if (obj) {
          Agent* agent = dynamic_cast<Agent*>(obj);
          if (agent) {
            agents.push_back(agent);
          }
        }
      }
    }

    return agents;
  }

  // Check if agents have sufficient resources for the given recipe
  bool can_afford_recipe(const Recipe& recipe, const std::vector<Agent*>& surrounding_agents) const {
    std::map<InventoryItem, InventoryQuantity> total_resources;
    for (Agent* agent : surrounding_agents) {
      for (const auto& [item, amount] : agent->inventory) {
        total_resources[item] = static_cast<InventoryQuantity>(total_resources[item] + amount);
      }
    }
    for (const auto& [item, required_amount] : recipe.input_resources) {
      if (total_resources[item] < required_amount) {
        return false;
      }
    }
    return true;
  }

  // Consume resources from surrounding agents for the given recipe
  void consume_resources_for_recipe(const Recipe& recipe, const std::vector<Agent*>& surrounding_agents) {
    for (const auto& [item, required_amount] : recipe.input_resources) {
      InventoryQuantity remaining = required_amount;
      for (Agent* agent : surrounding_agents) {
        if (remaining == 0) break;
        auto it = agent->inventory.find(item);
        if (it != agent->inventory.end()) {
          InventoryQuantity available = it->second;
          InventoryQuantity to_consume = static_cast<InventoryQuantity>(std::min<int>(available, remaining));
          InventoryDelta delta = agent->update_inventory(item, static_cast<InventoryDelta>(-to_consume));
          InventoryQuantity actually_consumed = static_cast<InventoryQuantity>(-delta);
          remaining = static_cast<InventoryQuantity>(remaining - actually_consumed);
          if (actually_consumed > 0) {
            stats.add(stats.resource_name(item) + ".consumed", actually_consumed);
          }
        }
      }
    }
  }

  // Give output resources to the triggering agent
  void give_output_to_agent(const Recipe& recipe, Agent* agent) {
    for (const auto& [item, amount] : recipe.output_resources) {
      InventoryDelta delta = agent->update_inventory(item, static_cast<InventoryDelta>(amount));
      InventoryQuantity actually_produced = static_cast<InventoryQuantity>(delta);
      if (actually_produced > 0) {
        stats.add(stats.resource_name(item) + ".produced", actually_produced);
      }
    }
  }

public:
  // Recipe lookup table - 256 possible patterns (2^8)
  std::vector<std::shared_ptr<Recipe>> recipes;

  // Current cooldown state
  bool cooling_down;
  unsigned short cooldown_remaining;

  // Event manager for scheduling cooldown events
  class EventManager* event_manager;

  // Stats tracking
  class StatsTracker stats;

  // Grid access for finding surrounding agents
  class Grid* grid;

  NanoAssembler(GridCoord r, GridCoord c, const NanoAssemblerConfig& cfg)
      : recipes(cfg.recipes), cooling_down(false), cooldown_remaining(0), event_manager(nullptr), grid(nullptr) {
    GridObject::init(cfg.type_id, cfg.type_name, GridLocation(r, c, GridLayer::ObjectLayer));
  }
  virtual ~NanoAssembler() = default;

  // Set event manager for cooldown scheduling
  void set_event_manager(class EventManager* event_manager_ptr) {
    this->event_manager = event_manager_ptr;
  }

  // Set grid access
  void set_grid(class Grid* grid_ptr) {
    this->grid = grid_ptr;
  }

  // Implement pure virtual method from Usable
  virtual bool onUse(Agent* actor, ActionArg /*arg*/) override {
    if (!grid || !event_manager) {
      return false;
    }
    if (cooling_down) {
      stats.incr("nano_assembler.blocked.cooldown");
      return false;
    }
    uint8_t pattern = get_agent_pattern_byte();
    Recipe* recipe = recipes[pattern].get();
    if (!recipe || (recipe->input_resources.empty() && recipe->output_resources.empty())) {
      stats.incr("nano_assembler.blocked.no_recipe");
      return false;
    }
    std::vector<Agent*> surrounding_agents = get_surrounding_agents();
    if (!can_afford_recipe(*recipe, surrounding_agents)) {
      stats.incr("nano_assembler.blocked.insufficient_resources");
      return false;
    }
    consume_resources_for_recipe(*recipe, surrounding_agents);
    give_output_to_agent(*recipe, actor);
    stats.incr("nano_assembler.recipes_executed");
    stats.incr("nano_assembler.recipe_pattern_" + std::to_string(pattern));
    if (recipe->cooldown > 0) {
      cooling_down = true;
      cooldown_remaining = recipe->cooldown;
      event_manager->schedule_event(EventType::CoolDown, recipe->cooldown, id, 0);
      stats.incr("nano_assembler.cooldown_started");
    }
    return true;
  }

  virtual std::vector<PartialObservationToken> obs_features() const override {
    std::vector<PartialObservationToken> features;
    features.push_back({ObservationFeature::TypeId, static_cast<ObservationType>(this->type_id)});
    features.push_back({ObservationFeature::ConvertingOrCoolingDown, static_cast<ObservationType>(this->cooling_down)});
    // features.push_back({ObservationFeature::Color, static_cast<ObservationType>(this->cooldown_remaining)});
    // uint8_t pattern = get_agent_pattern_byte();
    // features.push_back({ObservationFeature::Group, static_cast<ObservationType>(pattern)});
    return features;
  }

  // Handle cooldown completion
  void finish_cooldown() {
    this->cooling_down = false;
    this->cooldown_remaining = 0;
    stats.incr("nano_assembler.cooldown_completed");
  }
};

#endif  // OBJECTS_NANO_ASSEMBLER_HPP_
