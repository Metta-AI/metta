#ifndef PACKAGES_METTAGRID_CPP_INCLUDE_METTAGRID_OBJECTS_ASSEMBLER_HPP_
#define PACKAGES_METTAGRID_CPP_INCLUDE_METTAGRID_OBJECTS_ASSEMBLER_HPP_

#include <map>
#include <stdexcept>
#include <string>
#include <vector>

#include "core/event.hpp"
#include "core/grid.hpp"
#include "core/grid_object.hpp"
#include "core/types.hpp"
#include "objects/agent.hpp"
#include "objects/assembler_config.hpp"
#include "objects/constants.hpp"
#include "objects/recipe.hpp"
#include "objects/usable.hpp"
#include "systems/stats_tracker.hpp"

// Forward declaration
class Agent;

class Assembler : public GridObject, public Usable {
private:
  // Surrounding positions in deterministic order: NW, N, NE, W, E, SW, S, SE
  std::vector<std::pair<GridCoord, GridCoord>> get_surrounding_positions() const {
    GridCoord r = location.r;
    GridCoord c = location.c;
    std::vector<std::pair<GridCoord, GridCoord>> positions;
    for (int i = -1; i <= 1; ++i) {
      for (int j = -1; j <= 1; ++j) {
        if (i == 0 && j == 0) continue;  // skip center
        GridLocation position = {static_cast<GridCoord>(r + i), static_cast<GridCoord>(c + j)};
        if (grid->is_valid_location(position)) {
          positions.emplace_back(static_cast<GridCoord>(r + i), static_cast<GridCoord>(c + j));
        }
      }
    }
    return positions;
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
  void give_output_to_agent(const Recipe& recipe, Agent& agent) {
    for (const auto& [item, amount] : recipe.output_resources) {
      InventoryDelta delta = agent.update_inventory(item, static_cast<InventoryDelta>(amount));
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

  Assembler(GridCoord r, GridCoord c, const AssemblerConfig& cfg)
      : recipes(cfg.recipes), cooling_down(false), cooldown_remaining(0), event_manager(nullptr), grid(nullptr) {
    GridObject::init(cfg.type_id, cfg.type_name, GridLocation(r, c, GridLayer::ObjectLayer), cfg.tag_ids);
  }
  virtual ~Assembler() = default;

  // Set event manager for cooldown scheduling
  void set_event_manager(class EventManager* event_manager_ptr) {
    this->event_manager = event_manager_ptr;
  }

  // Set grid access
  void set_grid(class Grid* grid_ptr) {
    this->grid = grid_ptr;
  }

  // Implement pure virtual method from Usable
  virtual bool onUse(Agent& actor, ActionArg /*arg*/) override {
    if (!grid || !event_manager) {
      return false;
    }
    if (cooling_down) {
      stats.incr("assembler.blocked.cooldown");
      return false;
    }
    uint8_t pattern = get_agent_pattern_byte();
    Recipe* recipe = recipes[pattern].get();
    if (!recipe || (recipe->input_resources.empty() && recipe->output_resources.empty())) {
      stats.incr("assembler.blocked.no_recipe");
      return false;
    }
    std::vector<Agent*> surrounding_agents = get_surrounding_agents();
    if (!can_afford_recipe(*recipe, surrounding_agents)) {
      stats.incr("assembler.blocked.insufficient_resources");
      return false;
    }
    consume_resources_for_recipe(*recipe, surrounding_agents);
    give_output_to_agent(*recipe, actor);
    stats.incr("assembler.recipes_executed");
    stats.incr("assembler.recipe_pattern_" + std::to_string(pattern));
    if (recipe->cooldown > 0) {
      cooling_down = true;
      cooldown_remaining = recipe->cooldown;
      event_manager->schedule_event(EventType::CoolDown, recipe->cooldown, id, 0);
      stats.incr("assembler.cooldown_started");
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

    // Emit tag features
    for (int tag_id : this->tag_ids) {
      features.push_back({ObservationFeature::Tag, static_cast<ObservationType>(tag_id)});
    }

    return features;
  }

  // Handle cooldown completion
  void finish_cooldown() {
    this->cooling_down = false;
    this->cooldown_remaining = 0;
    stats.incr("assembler.cooldown_completed");
  }
};

#endif  // PACKAGES_METTAGRID_CPP_INCLUDE_METTAGRID_OBJECTS_ASSEMBLER_HPP_
