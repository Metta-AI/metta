#ifndef PACKAGES_METTAGRID_CPP_INCLUDE_METTAGRID_OBJECTS_ASSEMBLER_HPP_
#define PACKAGES_METTAGRID_CPP_INCLUDE_METTAGRID_OBJECTS_ASSEMBLER_HPP_

#include <algorithm>
#include <cassert>
#include <set>
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
  // Surrounding positions in deterministic order around the full assembler footprint.
  std::vector<std::pair<GridCoord, GridCoord>> get_surrounding_positions() const {
    std::vector<std::pair<GridCoord, GridCoord>> positions;
    if (!grid) return positions;

    // Track assembler-occupied cells so we can exclude them from the perimeter.
    std::set<std::pair<GridCoord, GridCoord>> assembler_cells;
    for (const auto& loc : locations) {
      assembler_cells.emplace(loc.r, loc.c);
    }

    std::set<std::pair<GridCoord, GridCoord>> unique_positions;
    for (const auto& loc : locations) {
      for (int dr = -1; dr <= 1; ++dr) {
        for (int dc = -1; dc <= 1; ++dc) {
          if (dr == 0 && dc == 0) continue;
          GridCoord nr = static_cast<GridCoord>(loc.r + dr);
          GridCoord nc = static_cast<GridCoord>(loc.c + dc);
          GridLocation neighbor(nr, nc);
          if (!grid->is_valid_location(neighbor)) continue;
          std::pair<GridCoord, GridCoord> cell(nr, nc);
          if (assembler_cells.count(cell) > 0) continue;
          unique_positions.insert(cell);
        }
      }
    }

    positions.reserve(unique_positions.size());
    for (const auto& cell : unique_positions) {
      positions.push_back(cell);
    }

    return positions;
  }

  // Get surrounding agents in deterministic upper-left-to-lower-right order.
  std::vector<Agent*> get_surrounding_agents() const {
    std::vector<Agent*> agents;
    if (!grid) return agents;

    std::vector<std::pair<GridCoord, GridCoord>> positions = get_surrounding_positions();

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
  bool static can_afford_protocol(const Protocol& protocol, const std::vector<Agent*>& surrounding_agents) {
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

  // Check if surrounding agents can receive output from the given protocol
  // Returns true if either (a) the protocol has no output, or (b) the surrounding agents
  // can absorb at least one item in the output. Returns false if the protocol produces
  // output and the surrounding agents cannot absorb any of it.
  bool static can_receive_output(const Protocol& protocol, const std::vector<Agent*>& surrounding_agents) {
    // If protocol has no positive output, return true
    if (!Assembler::protocol_has_positive_output(protocol)) {
      return true;
    }

    // If there are no surrounding agents, they can't absorb anything
    if (surrounding_agents.empty()) {
      return false;
    }

    // Check if agents can absorb at least one item for each output resource
    for (const auto& [item, amount] : protocol.output_resources) {
      if (amount > 0) {
        // Sum up free space across all surrounding agents for this item
        InventoryQuantity total_free_space = 0;
        for (Agent* agent : surrounding_agents) {
          total_free_space += agent->inventory.free_space(item);
        }
        // If at least one item can be absorbed, return true
        if (total_free_space >= 1) {
          return true;
        }
      }
    }

    // Protocol produces output but agents can absorb none of it
    return false;
  }

  // Give output resources to agents
  void static give_output_for_protocol(const Protocol& protocol, const std::vector<Agent*>& surrounding_agents) {
    std::vector<HasInventory*> agents_as_inventory_havers;
    for (Agent* agent : surrounding_agents) {
      agents_as_inventory_havers.push_back(static_cast<HasInventory*>(agent));
    }
    for (const auto& [item, amount] : protocol.output_resources) {
      HasInventory::shared_update(agents_as_inventory_havers, item, amount);
    }
  }

  // Returns true if the protocol yields any positive output amount
  bool static protocol_has_positive_output(const Protocol& protocol) {
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
  void static consume_resources_for_protocol(const Protocol& protocol, const std::vector<Agent*>& surrounding_agents) {
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
  const std::unordered_map<GroupVibe, std::vector<std::shared_ptr<Protocol>>> protocols;

  // Unclip protocols - used when assembler is clipped
  std::unordered_map<GroupVibe, std::vector<std::shared_ptr<Protocol>>> unclip_protocols;

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

  // Grid access for finding surrounding agents
  class Grid* grid;

  // Clipper pointer, for when we become unclipped.
  Clipper* clipper_ptr;

  // Pointer to current timestep from environment
  unsigned int* current_timestep_ptr;

  const class ObservationEncoder* obs_encoder;

  // Allow partial usage during cooldown
  bool allow_partial_usage;

  Assembler(GridCoord r, GridCoord c, const AssemblerConfig& cfg)
      : GridObject(cfg.type_id,
                   cfg.type_name,
                   std::vector<GridLocation>{GridLocation(r, c)},
                   cfg.tag_ids,
                   cfg.initial_vibe),
        protocols(build_protocol_map(cfg.protocols)),
        unclip_protocols(),
        is_clipped(cfg.start_clipped),
        clip_immune(cfg.clip_immune),
        start_clipped(cfg.start_clipped),
        cooldown_end_timestep(0),
        cooldown_duration(0),
        uses_count(0),
        max_uses(cfg.max_uses),
        grid(nullptr),
        clipper_ptr(nullptr),
        current_timestep_ptr(nullptr),
        obs_encoder(nullptr),
        allow_partial_usage(cfg.allow_partial_usage) {}
  virtual ~Assembler() = default;

  // Assemblers support multi-cell footprints (e.g., 2x2)
  bool supports_multi_cell() const override {
    return true;
  }

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
  static std::unordered_map<GroupVibe, std::vector<std::shared_ptr<Protocol>>> build_protocol_map(
      const std::vector<std::shared_ptr<Protocol>>& protocol_list) {
    std::unordered_map<GroupVibe, std::vector<std::shared_ptr<Protocol>>> protocol_map;
    for (const auto& protocol : protocol_list) {
      GroupVibe vibe = calculate_group_vibe_from_vibes(protocol->vibes);
      protocol_map[vibe].push_back(protocol);
    }
    for (auto& [vibe, protocol_list] : protocol_map) {
      std::sort(protocol_list.begin(),
                protocol_list.end(),
                [](const std::shared_ptr<Protocol>& a, const std::shared_ptr<Protocol>& b) {
                  return a->min_agents > b->min_agents;
                });
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
    size_t num_agents = get_surrounding_agents().size();

    auto protocols_to_use = protocols;
    if (is_clipped) {
      protocols_to_use = unclip_protocols;
    }

    auto it = protocols_to_use.find(vibe);
    if (it != protocols_to_use.end()) {
      for (const auto& protocol : it->second) {
        if (protocol->min_agents <= num_agents) {
          return protocol.get();
        }
      }
    }

    // Check the default if no protocol is found for the current vibe.
    it = protocols_to_use.find(0);
    if (it != protocols_to_use.end()) {
      for (const auto& protocol : it->second) {
        if (protocol->min_agents <= num_agents) {
          return protocol.get();
        }
      }
    }

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
      if (!Assembler::protocol_has_positive_output(protocol_to_use) &&
          Assembler::protocol_has_positive_output(*original_protocol) && !is_clipped) {
        return false;
      }
    }

    std::vector<Agent*> surrounding_agents = get_surrounding_agents();
    // Always include the acting agent in the resource-sharing neighborhood.
    if (std::find(surrounding_agents.begin(), surrounding_agents.end(), &actor) == surrounding_agents.end()) {
      surrounding_agents.push_back(&actor);
    }
    if (!Assembler::can_afford_protocol(protocol_to_use, surrounding_agents)) {
      return false;
    }
    if (!Assembler::can_receive_output(protocol_to_use, surrounding_agents) && !is_clipped) {
      // If the agents gain nothing from the protocol, don't use it.
      return false;
    }

    consume_resources_for_protocol(protocol_to_use, surrounding_agents);
    give_output_for_protocol(protocol_to_use, surrounding_agents);

    cooldown_duration = static_cast<unsigned int>(protocol_to_use.cooldown);
    cooldown_end_timestep = *current_timestep_ptr + cooldown_duration;

    // If we were clipped and successfully used an unclip protocol, become unclipped. Also, don't count this as a use.
    if (is_clipped) {
      become_unclipped();
    } else {
      uses_count++;
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
    if (this->obs_encoder && this->obs_encoder->protocol_details_obs) {
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
