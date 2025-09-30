#ifndef PACKAGES_METTAGRID_CPP_INCLUDE_METTAGRID_OBJECTS_ASSEMBLER_HPP_
#define PACKAGES_METTAGRID_CPP_INCLUDE_METTAGRID_OBJECTS_ASSEMBLER_HPP_

#include <map>
#include <stdexcept>
#include <string>
#include <vector>

#include "core/grid.hpp"
#include "core/grid_object.hpp"
#include "core/types.hpp"
#include "objects/agent.hpp"
#include "objects/assembler_config.hpp"
#include "objects/constants.hpp"
#include "objects/recipe.hpp"
#include "objects/usable.hpp"

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
      for (const auto& [item, amount] : agent->inventory.get()) {
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

  // Give output resources to the triggering agent
  void give_output_to_agent(const Recipe& recipe, Agent& agent) {
    for (const auto& [item, amount] : recipe.output_resources) {
      agent.update_inventory(item, static_cast<InventoryDelta>(amount));
    }
  }

public:
  // Consume resources from surrounding agents for the given recipe
  // Intended to be private, but made public for testing. We couldn't get `friend` to work as expected.
  void consume_resources_for_recipe(const Recipe& recipe, const std::vector<Agent*>& surrounding_agents) {
    for (const auto& [item, required_amount] : recipe.input_resources) {
      InventoryQuantity remaining = required_amount;
      for (Agent* agent : surrounding_agents) {
        if (remaining == 0) break;
        InventoryQuantity available = agent->inventory.amount(item);
        InventoryQuantity to_consume = static_cast<InventoryQuantity>(std::min<int>(available, remaining));
        agent->update_inventory(item, static_cast<InventoryDelta>(-to_consume));
        remaining -= to_consume;
      }
    }
  }

  // Recipe lookup table - 256 possible patterns (2^8)
  std::vector<std::shared_ptr<Recipe>> recipes;

  // Unclip recipes - used when assembler is clipped
  std::vector<std::shared_ptr<Recipe>> unclip_recipes;

  // Clipped state
  bool is_clipped;

  // Current cooldown state
  unsigned int cooldown_end_timestep;

  // Usage tracking
  unsigned int max_uses;    // Maximum number of uses (0 = unlimited)
  unsigned int uses_count;  // Current number of times used

  // Grid access for finding surrounding agents
  class Grid* grid;

  // Pointer to current timestep from environment
  unsigned int* current_timestep_ptr;

  // Recipe observation configuration
  bool recipe_details_obs;
  ObservationType input_recipe_offset;
  ObservationType output_recipe_offset;

  Assembler(GridCoord r, GridCoord c, const AssemblerConfig& cfg)
      : recipes(cfg.recipes),
        unclip_recipes(),
        is_clipped(false),
        cooldown_end_timestep(0),
        max_uses(cfg.max_uses),
        uses_count(0),
        grid(nullptr),
        current_timestep_ptr(nullptr),
        recipe_details_obs(cfg.recipe_details_obs),
        input_recipe_offset(cfg.input_recipe_offset),
        output_recipe_offset(cfg.output_recipe_offset) {
    GridObject::init(cfg.type_id, cfg.type_name, GridLocation(r, c, GridLayer::ObjectLayer), cfg.tag_ids);
  }
  virtual ~Assembler() = default;

  // Set grid access
  void set_grid(class Grid* grid_ptr) {
    this->grid = grid_ptr;
  }

  // Set current timestep pointer
  void set_current_timestep_ptr(unsigned int* timestep_ptr) {
    this->current_timestep_ptr = timestep_ptr;
  }

  // Calculate remaining cooldown time
  unsigned int cooldown_remaining() const {
    if (!current_timestep_ptr || cooldown_end_timestep <= *current_timestep_ptr) {
      return 0;
    }
    return cooldown_end_timestep - *current_timestep_ptr;
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

  // Get current recipe based on surrounding agent pattern
  const Recipe* get_current_recipe() const {
    if (!grid) return nullptr;
    uint8_t pattern = get_agent_pattern_byte();

    // Use unclip recipes if clipped, normal recipes otherwise
    const std::vector<std::shared_ptr<Recipe>>& active_recipes = is_clipped ? unclip_recipes : recipes;

    if (pattern >= active_recipes.size()) return nullptr;
    return active_recipes[pattern].get();
  }

  // Make this assembler clipped with the given unclip recipes
  void becomeClipped(const std::vector<std::shared_ptr<Recipe>>& unclip_recipes_vec) {
    is_clipped = true;
    unclip_recipes = unclip_recipes_vec;
  }

  virtual bool onUse(Agent& actor, ActionArg /*arg*/) override {
    if (!grid || !current_timestep_ptr) {
      return false;
    }
    // Check if max uses has been reached
    if (max_uses > 0 && uses_count >= max_uses) {
      return false;
    }
    if (cooldown_remaining() > 0) {
      return false;
    }
    const Recipe* recipe = get_current_recipe();
    if (!recipe || (recipe->input_resources.empty() && recipe->output_resources.empty())) {
      return false;
    }
    std::vector<Agent*> surrounding_agents = get_surrounding_agents();
    if (!can_afford_recipe(*recipe, surrounding_agents)) {
      return false;
    }
    consume_resources_for_recipe(*recipe, surrounding_agents);
    give_output_to_agent(*recipe, actor);
    if (recipe->cooldown > 0) {
      cooldown_end_timestep = *current_timestep_ptr + recipe->cooldown;
    }

    // If we were clipped and successfully used an unclip recipe, become unclipped. Also, don't count this as a use.
    if (is_clipped) {
      is_clipped = false;
      unclip_recipes.clear();
    } else {
      uses_count++;
    }

    return true;
  }

  virtual std::vector<PartialObservationToken> obs_features() const override {
    std::vector<PartialObservationToken> features;
    features.push_back({ObservationFeature::TypeId, static_cast<ObservationType>(this->type_id)});

    unsigned int remaining = std::min(cooldown_remaining(), 255u);
    if (remaining > 0) {
      features.push_back({ObservationFeature::CooldownRemaining, static_cast<ObservationType>(remaining)});
    }

    // Add clipped status to observations if clipped
    if (is_clipped) {
      features.push_back({ObservationFeature::Clipped, static_cast<ObservationType>(1)});
    }

    // Add remaining uses to observations if max_uses is set
    if (max_uses > 0) {
      unsigned int remaining_uses = (uses_count < max_uses) ? (max_uses - uses_count) : 0;
      remaining_uses = std::min(remaining_uses, 255u);  // Cap at 255 for observation
      features.push_back({ObservationFeature::RemainingUses, static_cast<ObservationType>(remaining_uses)});
    }

    // Add recipe details if configured to do so
    if (this->recipe_details_obs) {
      const Recipe* current_recipe = get_current_recipe();
      if (current_recipe) {
        // Add recipe inputs (input:resource) - only non-zero values
        for (const auto& [item, amount] : current_recipe->input_resources) {
          if (amount > 0) {
            features.push_back(
                {static_cast<ObservationType>(input_recipe_offset + item), static_cast<ObservationType>(amount)});
          }
        }

        // Add recipe outputs (output:resource) - only non-zero values
        for (const auto& [item, amount] : current_recipe->output_resources) {
          if (amount > 0) {
            features.push_back(
                {static_cast<ObservationType>(output_recipe_offset + item), static_cast<ObservationType>(amount)});
          }
        }
      }
    }

    // Emit tag features
    for (int tag_id : this->tag_ids) {
      features.push_back({ObservationFeature::Tag, static_cast<ObservationType>(tag_id)});
    }

    return features;
  }
};

#endif  // PACKAGES_METTAGRID_CPP_INCLUDE_METTAGRID_OBJECTS_ASSEMBLER_HPP_
