#ifndef PACKAGES_METTAGRID_CPP_INCLUDE_METTAGRID_OBJECTS_ASSEMBLER_HPP_
#define PACKAGES_METTAGRID_CPP_INCLUDE_METTAGRID_OBJECTS_ASSEMBLER_HPP_

#include <algorithm>
#include <cassert>
#include <stdexcept>
#include <string>
#include <unordered_map>
#include <vector>

#include "config/observation_features.hpp"
#include "core/grid.hpp"
#include "core/grid_object.hpp"
#include "core/types.hpp"
#include "objects/agent.hpp"
#include "objects/assembler_config.hpp"
#include "objects/constants.hpp"
#include "objects/protocol.hpp"
#include "objects/usable.hpp"
#include "systems/observation_encoder.hpp"

class Clipper;

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
        GridObject* obj = grid->object_at(GridLocation(check_r, check_c));
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

  // Check if agents have sufficient resources for the given protocol
  bool can_afford_protocol(const Protocol& protocol, const std::vector<Agent*>& surrounding_agents) const {
    std::unordered_map<InventoryItem, InventoryQuantity> total_resources;
    for (Agent* agent : surrounding_agents) {
      for (const auto& [item, amount] : agent->inventory.get()) {
        total_resources[item] = static_cast<InventoryQuantity>(total_resources[item] + amount);
      }
    }
    for (const auto& [item, required_amount] : protocol.input_resources) {
      if (total_resources[item] < required_amount) {
        return false;
      }
    }
    return true;
  }

  // Give output resources to agents
  void give_output_for_protocol(const Protocol& protocol, const std::vector<Agent*>& surrounding_agents) {
    std::vector<HasInventory*> agents_as_inventory_havers;
    for (Agent* agent : surrounding_agents) {
      agents_as_inventory_havers.push_back(static_cast<HasInventory*>(agent));
    }
    for (const auto& [item, amount] : protocol.output_resources) {
      HasInventory::shared_update(agents_as_inventory_havers, item, amount);
    }
  }

  // Returns true if the protocol yields any positive output amount
  bool protocol_has_positive_output(const Protocol& protocol) const {
    for (const auto& [item, amount] : protocol.output_resources) {
      if (amount > 0) {
        return true;
      }
    }
    return false;
  }

public:
  // Consume resources from surrounding agents for the given protocol
  // Intended to be private, but made public for testing. We couldn't get `friend` to work as expected.
  void consume_resources_for_protocol(const Protocol& protocol, const std::vector<Agent*>& surrounding_agents) {
    std::vector<HasInventory*> agents_as_inventory_havers;
    for (Agent* agent : surrounding_agents) {
      agents_as_inventory_havers.push_back(static_cast<HasInventory*>(agent));
    }
    for (const auto& [item, required_amount] : protocol.input_resources) {
      InventoryDelta consumed = HasInventory::shared_update(agents_as_inventory_havers, item, -required_amount);
      assert(consumed == -required_amount && "Expected all required resources to be consumed");
    }
  }

  // Protocol lookup table for protocols that depend on agents vibing- keyed by local vibe (64-bit number from sorted
  // vibes). Later, this may be switched to having string keys based on the vibes.
  // Note that 0 is both the vibe you get when no one is showing a vibe, and also the default vibe.
  const std::unordered_map<GroupVibe, std::shared_ptr<Protocol>> protocols;

  // Unclip protocols - used when assembler is clipped
  std::unordered_map<GroupVibe, std::shared_ptr<Protocol>> unclip_protocols;

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

  // Protocol observation configuration
  bool protocol_details_obs;
  const class ObservationEncoder* obs_encoder;

  // Allow partial usage during cooldown
  bool allow_partial_usage;

  Assembler(GridCoord r, GridCoord c, const AssemblerConfig& cfg)
      : protocols(build_protocol_map(cfg.protocols)),
        unclip_protocols(),
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
        protocol_details_obs(cfg.protocol_details_obs),
        allow_partial_usage(cfg.allow_partial_usage),
        clipper_ptr(nullptr) {
    GridObject::init(cfg.type_id, cfg.type_name, GridLocation(r, c), cfg.tag_ids, cfg.initial_vibe);
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

  // Set observation encoder for protocol feature ID lookup
  void set_obs_encoder(const class ObservationEncoder* encoder) {
    this->obs_encoder = encoder;
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

  // Helper function to calculate GroupVibe from a vector of glyphs
  static GroupVibe calculate_group_vibe_from_vibes(std::vector<uint8_t> vibes) {
    // Sort the glyphs to make the vibe independent of agent positions.
    std::sort(vibes.begin(), vibes.end());
    return std::accumulate(
        vibes.begin(), vibes.end(), GroupVibe{0}, [](GroupVibe acc, uint8_t vibe) { return (acc << 8) | vibe; });
  }

  // Helper function to build a protocol map from a vector of protocols
  static std::unordered_map<GroupVibe, std::shared_ptr<Protocol>> build_protocol_map(
      const std::vector<std::shared_ptr<Protocol>>& protocol_list) {
    std::unordered_map<GroupVibe, std::shared_ptr<Protocol>> protocol_map;
    for (const auto& protocol : protocol_list) {
      GroupVibe vibe = calculate_group_vibe_from_vibes(protocol->vibes);
      protocol_map[vibe] = protocol;
    }
    return protocol_map;
  }

  // Helper function to get the "local vibe" based on vibes of surrounding agents
  // Returns a 64-bit number created from sorted vibes of surrounding agents
  GroupVibe get_local_vibe() const {
    if (!grid) return 0;

    std::vector<uint8_t> vibes;
    std::vector<std::pair<GridCoord, GridCoord>> positions = get_surrounding_positions();

    for (size_t i = 0; i < positions.size(); i++) {
      GridCoord check_r = positions[i].first;
      GridCoord check_c = positions[i].second;

      if (check_r < grid->height && check_c < grid->width) {
        GridObject* obj = grid->object_at(GridLocation(check_r, check_c));
        if (obj) {
          Agent* agent = dynamic_cast<Agent*>(obj);
          if (agent && agent->vibe != 0) {
            vibes.push_back(agent->vibe);
          }
        }
      }
    }

    return calculate_group_vibe_from_vibes(vibes);
  }

  // Get current protocol based on local vibe from surrounding agent vibes
  const Protocol* get_current_protocol() const {
    if (!grid) return nullptr;
    GroupVibe vibe = get_local_vibe();

    auto protocols_to_use = protocols;
    if (is_clipped) {
      protocols_to_use = unclip_protocols;
    }

    auto it = protocols_to_use.find(vibe);
    if (it != protocols_to_use.end()) return it->second.get();

    // Check the default if no protocol is found for the current vibe.
    it = protocols_to_use.find(0);
    if (it != protocols_to_use.end()) return it->second.get();

    return nullptr;
  }

  // Make this assembler clipped with the given unclip protocols
  void become_clipped(const std::vector<std::shared_ptr<Protocol>>& unclip_protocols_list, Clipper* clipper) {
    is_clipped = true;
    unclip_protocols = build_protocol_map(unclip_protocols_list);
    // It's a little odd that we store the clipper here, versus having global access to it. This is a
    // path of least resistance, not a specific intention. But it does present questions around whether
    // there could be more than one Clipper.
    clipper_ptr = clipper;
    // Reset cooldown. The assembler being on its normal cooldown shouldn't stop it from being unclipped.
    cooldown_end_timestep = *current_timestep_ptr;
    cooldown_duration = 0;
  }

  void become_unclipped();

  // Scale protocol requirements based on cooldown progress (for partial usage)
  const Protocol scale_protocol_for_partial_usage(const Protocol& original_protocol, float progress) const {
    Protocol scaled_protocol;

    // Scale input resources (multiply by progress and round up)
    for (const auto& [resource, amount] : original_protocol.input_resources) {
      InventoryQuantity scaled_amount = static_cast<InventoryQuantity>(std::ceil(amount * progress));
      scaled_protocol.input_resources[resource] = scaled_amount;
    }

    // Scale output resources (multiply by progress and round down)
    for (const auto& [resource, amount] : original_protocol.output_resources) {
      InventoryQuantity scaled_amount = static_cast<InventoryQuantity>(std::floor(amount * progress));
      scaled_protocol.output_resources[resource] = scaled_amount;
    }

    // Keep the same cooldown and vibes
    scaled_protocol.cooldown = original_protocol.cooldown;
    scaled_protocol.vibes = original_protocol.vibes;

    return scaled_protocol;
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

    const Protocol* original_protocol = get_current_protocol();
    if (!original_protocol) {
      return false;
    }

    Protocol protocol_to_use = *original_protocol;
    if (progress < 1.0f && allow_partial_usage) {
      protocol_to_use = scale_protocol_for_partial_usage(*original_protocol, progress);

      // Prevent usage that would yield no outputs (and would only serve to burn inputs and increment uses_count)
      // Do not prevent usage if:
      // - the unscaled protocol does not have outputs
      // - usage would unclip the assembler; the unscaled unclipping protocol may happen to include outputs
      if (!protocol_has_positive_output(protocol_to_use) && protocol_has_positive_output(*original_protocol) &&
          !is_clipped) {
        return false;
      }
    }

    std::vector<Agent*> surrounding_agents = get_surrounding_agents(&actor);
    if (!can_afford_protocol(protocol_to_use, surrounding_agents)) {
      return false;
    }
    consume_resources_for_protocol(protocol_to_use, surrounding_agents);
    give_output_for_protocol(protocol_to_use, surrounding_agents);

    cooldown_duration = static_cast<unsigned int>(protocol_to_use.cooldown * cooldown_multiplier);
    cooldown_end_timestep = *current_timestep_ptr + cooldown_duration;

    // If we were clipped and successfully used an unclip protocol, become unclipped. Also, don't count this as a use.
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

    // Add protocol details if configured to do so
    if (this->protocol_details_obs && this->obs_encoder) {
      const Protocol* current_protocol = get_current_protocol();
      if (current_protocol) {
        // Add protocol inputs (input:resource) - only non-zero values
        for (const auto& [item, amount] : current_protocol->input_resources) {
          if (amount > 0) {
            features.push_back({obs_encoder->get_input_feature_id(item), static_cast<ObservationType>(amount)});
          }
        }

        // Add protocol outputs (output:resource) - only non-zero values
        for (const auto& [item, amount] : current_protocol->output_resources) {
          if (amount > 0) {
            features.push_back({obs_encoder->get_output_feature_id(item), static_cast<ObservationType>(amount)});
          }
        }
      }
    }

    // Emit tag features
    for (int tag_id : this->tag_ids) {
      features.push_back({ObservationFeature::Tag, static_cast<ObservationType>(tag_id)});
    }

    if (this->vibe != 0) {
      features.push_back({ObservationFeature::Vibe, static_cast<ObservationType>(this->vibe)});
    }

    return features;
  }
};

#include "systems/clipper.hpp"

inline void Assembler::become_unclipped() {
  is_clipped = false;
  unclip_protocols.clear();
  if (clipper_ptr) {
    // clipper_ptr might not be set if we're being unclipped as part of a test.
    // Later, it might be because we started clipped.
    clipper_ptr->on_unclip_assembler(*this);
  }
  clipper_ptr = nullptr;
}

#endif  // PACKAGES_METTAGRID_CPP_INCLUDE_METTAGRID_OBJECTS_ASSEMBLER_HPP_
