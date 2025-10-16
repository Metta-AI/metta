#ifndef PACKAGES_METTAGRID_CPP_INCLUDE_METTAGRID_OBJECTS_ASSEMBLER_HPP_
#define PACKAGES_METTAGRID_CPP_INCLUDE_METTAGRID_OBJECTS_ASSEMBLER_HPP_

#include <algorithm>
#include <cassert>
#include <stdexcept>
#include <string>
#include <unordered_map>
#include <vector>

#include "core/grid.hpp"
#include "core/grid_object.hpp"
#include "core/types.hpp"
#include "objects/agent.hpp"
#include "objects/assembler_config.hpp"
#include "objects/constants.hpp"
#include "objects/recipe.hpp"
#include "objects/usable.hpp"

class Clipper;

class Assembler : public GridObject, public Usable {
private:
  // Surrounding positions in deterministic order: NW, N, NE, W, E, SW, S, SE
  // This order is important for get_agent_pattern_byte which uses bit positions
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

  // Get surrounding agents in upper-left-to-lower-right order starting from the given agent's position
  std::vector<Agent*> get_surrounding_agents(const Agent* starting_agent) const {
    std::vector<Agent*> agents;
    if (!grid) return agents;

    std::vector<std::pair<GridCoord, GridCoord>> positions = get_surrounding_positions();

    // Find the starting agent's position in the surrounding positions
    int start_index = -1;
    if (starting_agent) {
      for (size_t i = 0; i < positions.size(); i++) {
        if (positions[i].first == starting_agent->location.r && positions[i].second == starting_agent->location.c) {
          start_index = i;
          break;
        }
      }

      // The starting agent must be in one of the surrounding positions
      if (start_index == -1) {
        throw std::runtime_error("Starting agent is not in a surrounding position of the assembler");
      }
    }

    // If starting agent was found in surrounding positions, reorder to start from there
    if (start_index >= 0) {
      // Rotate the positions vector to start from the starting_agent's position
      std::rotate(positions.begin(), positions.begin() + start_index, positions.end());
    }

    // Collect agents from the reordered positions
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
    std::unordered_map<InventoryItem, InventoryQuantity> total_resources;
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

  // Give output resources to agents
  void give_output_for_recipe(const Recipe& recipe, const std::vector<Agent*>& surrounding_agents) {
    std::vector<Inventory*> inventories;
    for (Agent* agent : surrounding_agents) {
      inventories.push_back(&agent->inventory);
    }
    for (const auto& [item, amount] : recipe.output_resources) {
      Inventory::shared_update(inventories, item, amount);
    }
  }

  // Returns true if the recipe yields any positive output amount (legacy name retained for compatibility)
  bool recipe_has_positive_output(const Recipe& recipe) const {
    for (const auto& [item, amount] : recipe.output_resources) {
      if (amount > 0) {
        return true;
      }
    }
    return false;
  }

public:
  // Consume resources from surrounding agents for the given recipe
  // Intended to be private, but made public for testing. We couldn't get `friend` to work as expected.
  void consume_resources_for_recipe(const Recipe& recipe, const std::vector<Agent*>& surrounding_agents) {
    std::vector<Inventory*> inventories;
    for (Agent* agent : surrounding_agents) {
      inventories.push_back(&agent->inventory);
    }
    for (const auto& [item, required_amount] : recipe.input_resources) {
      InventoryDelta consumed = Inventory::shared_update(inventories, item, -required_amount);
      assert(consumed == -required_amount && "Expected all required resources to be consumed");
    }
  }

  // Recipe lookup table - 256 possible patterns (2^8)
  std::vector<std::shared_ptr<Recipe>> recipes;

  // Unclip recipes - used when assembler is clipped
  std::vector<std::shared_ptr<Recipe>> unclip_recipes;

  // Clipped state
  bool is_clipped;

  // Clip immunity - if true, cannot be clipped
  bool clip_immune;

  // Start clipped - if true, starts in clipped state
  bool start_clipped;

  // Current cooldown state
  unsigned int cooldown_end_timestep;
  unsigned int cooldown_duration;  // Total duration of current cooldown

  // Usage tracking
  unsigned int uses_count;  // Current number of times used
  unsigned int max_uses;    // Maximum number of uses (0 = unlimited)

  // Exhaustion tracking
  float exhaustion;           // Exhaustion rate (0 = no exhaustion)
  float cooldown_multiplier;  // Current cooldown multiplier from exhaustion

  // Grid access for finding surrounding agents
  class Grid* grid;

  // Clipper pointer, for when we become unclipped.
  Clipper* clipper_ptr;

  // Pointer to current timestep from environment
  unsigned int* current_timestep_ptr;

  // Recipe observation configuration
  bool recipe_details_obs;
  ObservationType input_recipe_offset;
  ObservationType output_recipe_offset;

  // Allow partial usage during cooldown
  bool allow_partial_usage;

  Assembler(GridCoord r, GridCoord c, const AssemblerConfig& cfg)
      : recipes(cfg.recipes),
        unclip_recipes(),
        is_clipped(false),
        clip_immune(cfg.clip_immune),
        start_clipped(cfg.start_clipped),
        cooldown_end_timestep(0),
        cooldown_duration(0),
        uses_count(0),
        max_uses(cfg.max_uses),
        exhaustion(cfg.exhaustion),
        cooldown_multiplier(1.0f),
        grid(nullptr),
        current_timestep_ptr(nullptr),
        recipe_details_obs(cfg.recipe_details_obs),
        input_recipe_offset(cfg.input_recipe_offset),
        output_recipe_offset(cfg.output_recipe_offset),
        allow_partial_usage(cfg.allow_partial_usage),
        clipper_ptr(nullptr) {
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

  // Get the remaining cooldown duration in ticks (0 when ready for use)
  unsigned int cooldown_remaining() const {
    if (!current_timestep_ptr || cooldown_end_timestep <= *current_timestep_ptr) {
      return 0;
    }
    return cooldown_end_timestep - *current_timestep_ptr;
  }

  float cooldown_progress() const {
    // If no cooldown is active or no timestep pointer, return 1.0 (completed)
    if (!current_timestep_ptr || cooldown_duration == 0 || cooldown_end_timestep <= *current_timestep_ptr) {
      return 1.0f;
    }

    // Calculate how much time has elapsed since cooldown started
    unsigned int cooldown_start = cooldown_end_timestep - cooldown_duration;
    if (*current_timestep_ptr <= cooldown_start) {
      return 0.0f;  // Cooldown just started
    }

    unsigned int elapsed = *current_timestep_ptr - cooldown_start;
    return static_cast<float>(elapsed) / static_cast<float>(cooldown_duration);
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
  void become_clipped(const std::vector<std::shared_ptr<Recipe>>& unclip_recipes_vec, Clipper* clipper) {
    is_clipped = true;
    unclip_recipes = unclip_recipes_vec;
    // It's a little odd that we store the clipper here, versus having global access to it. This is a
    // path of least resistance, not a specific intention. But it does present questions around whether
    // there could be more than one Clipper.
    clipper_ptr = clipper;
    // Reset cooldown. The assembler being on its normal cooldown shouldn't stop it from being unclipped.
    cooldown_end_timestep = *current_timestep_ptr;
    cooldown_duration = 0;
  }

  void become_unclipped();

  // Scale recipe requirements based on cooldown progress (for partial usage)
  const Recipe scale_recipe_for_partial_usage(const Recipe& original_recipe, float progress) const {
    Recipe scaled_recipe;

    // Scale input resources (multiply by progress and round up)
    for (const auto& [resource, amount] : original_recipe.input_resources) {
      InventoryQuantity scaled_amount = static_cast<InventoryQuantity>(std::ceil(amount * progress));
      scaled_recipe.input_resources[resource] = scaled_amount;
    }

    // Scale output resources (multiply by progress and round down)
    for (const auto& [resource, amount] : original_recipe.output_resources) {
      InventoryQuantity scaled_amount = static_cast<InventoryQuantity>(std::floor(amount * progress));
      scaled_recipe.output_resources[resource] = scaled_amount;
    }

    // Keep the same cooldown
    scaled_recipe.cooldown = original_recipe.cooldown;

    return scaled_recipe;
  }

  virtual bool onUse(Agent& actor, ActionArg /*arg*/) override {
    if (!grid || !current_timestep_ptr) {
      return false;
    }

    if (max_uses > 0 && uses_count >= max_uses) {
      return false;
    }

    // Check if on cooldown and whether partial usage is allowed
    float progress = cooldown_progress();
    if (progress < 1.0f && !allow_partial_usage) {
      return false;  // On cooldown and partial usage not allowed
    }

    const Recipe* original_recipe = get_current_recipe();
    if (!original_recipe) {
      return false;
    }

    Recipe recipe_to_use = *original_recipe;
    if (progress < 1.0f && allow_partial_usage) {
      recipe_to_use = scale_recipe_for_partial_usage(*original_recipe, progress);

      // Prevent usage that would yield no outputs (and would only serve to burn inputs and increment uses_count)
      // Do not prevent usage if:
      // - the unscaled recipe does not have outputs
      // - usage would unclip the assembler; the unscaled unclipping recipe may happen to include outputs
      if (!recipe_has_positive_output(recipe_to_use) && recipe_has_positive_output(*original_recipe) && !is_clipped) {
        return false;
      }
    }

    std::vector<Agent*> surrounding_agents = get_surrounding_agents(&actor);
    if (!can_afford_recipe(recipe_to_use, surrounding_agents)) {
      return false;
    }
    consume_resources_for_recipe(recipe_to_use, surrounding_agents);
    give_output_for_recipe(recipe_to_use, surrounding_agents);

    cooldown_duration = static_cast<unsigned int>(recipe_to_use.cooldown * cooldown_multiplier);
    cooldown_end_timestep = *current_timestep_ptr + cooldown_duration;

    // If we were clipped and successfully used an unclip recipe, become unclipped. Also, don't count this as a use.
    if (is_clipped) {
      become_unclipped();
    } else {
      uses_count++;

      // Apply exhaustion (increase cooldown multiplier exponentially)
      if (exhaustion > 0.0f) {
        cooldown_multiplier *= (1.0f + exhaustion);
      }
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

#include "systems/clipper.hpp"

inline void Assembler::become_unclipped() {
  is_clipped = false;
  unclip_recipes.clear();
  if (clipper_ptr) {
    // clipper_ptr might not be set if we're being unclipped as part of a test.
    // Later, it might be because we started clipped.
    clipper_ptr->on_unclip_assembler(*this);
  }
  clipper_ptr = nullptr;
}

#endif  // PACKAGES_METTAGRID_CPP_INCLUDE_METTAGRID_OBJECTS_ASSEMBLER_HPP_
