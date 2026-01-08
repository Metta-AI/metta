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
#include "objects/chest.hpp"
#include "objects/constants.hpp"
#include "objects/inventory.hpp"
#include "objects/protocol.hpp"
#include "objects/usable.hpp"
#include "systems/observation_encoder.hpp"
#include "systems/stats_tracker.hpp"

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

  // Get surrounding chests within chest_search_distance (Chebyshev distance)
  // Returns empty vector if chest_search_distance is 0
  std::vector<Chest*> get_surrounding_chests() const {
    std::vector<Chest*> chests;
    if (!grid || chest_search_distance == 0) return chests;

    GridCoord r = location.r;
    GridCoord c = location.c;

    for (int dr = -static_cast<int>(chest_search_distance); dr <= static_cast<int>(chest_search_distance); ++dr) {
      for (int dc = -static_cast<int>(chest_search_distance); dc <= static_cast<int>(chest_search_distance); ++dc) {
        // Skip center (assembler itself)
        if (dr == 0 && dc == 0) continue;

        GridCoord check_r = r + dr;
        GridCoord check_c = c + dc;
        GridLocation check_location(check_r, check_c);

        if (grid->is_valid_location(check_location)) {
          if (Chest* chest = dynamic_cast<Chest*>(grid->object_at(check_location))) {
            chests.push_back(chest);
          }
        }
      }
    }

    return chests;
  }

  // Check if inventories have sufficient resources for the given protocol
  bool static can_afford_protocol(const Protocol& protocol, const std::vector<Inventory*>& surrounding_inventories) {
    std::unordered_map<InventoryItem, InventoryQuantity> total_resources;
    for (Inventory* inventory : surrounding_inventories) {
      for (const auto& [item, amount] : inventory->get()) {
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

  // Check if surrounding inventories can receive output from the given protocol
  // Returns true if either (a) the protocol has no output, or (b) the surrounding inventories
  // can absorb at least one item in the output. Returns false if the protocol produces
  // output and the surrounding inventories cannot absorb any of it.
  bool static can_receive_output(const Protocol& protocol, const std::vector<Inventory*>& surrounding_inventories) {
    // If protocol has no positive output, return true
    if (!Assembler::protocol_has_positive_output(protocol)) {
      return true;
    }

    // If there are no surrounding inventories, they can't absorb anything
    if (surrounding_inventories.empty()) {
      return false;
    }

    // Check if inventories can absorb at least one item
    for (const auto& [item, amount] : protocol.output_resources) {
      if (amount > 0) {
        // Sum up free space across all surrounding inventories for this item
        InventoryQuantity total_free_space = 0;
        for (Inventory* inventory : surrounding_inventories) {
          total_free_space += inventory->free_space(item);
        }
        // If at least one item can be absorbed, return true
        if (total_free_space >= 1) {
          return true;
        }
      }
    }

    // Protocol produces output but inventories can absorb none of it
    return false;
  }

  // Give output resources to inventories and log creation stats
  void give_output_for_protocol(const Protocol& protocol, const std::vector<Inventory*>& surrounding_inventories) {
    for (const auto& [item, amount] : protocol.output_resources) {
      InventoryDelta distributed = HasInventory::shared_update(surrounding_inventories, item, amount);

      // Count newly created outputs
      if (stats_tracker && distributed > 0) {
        const std::string& resource_name = stats_tracker->resource_name(item);
        stats_tracker->add("assembler." + resource_name + ".created", static_cast<float>(distributed));
      }
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

  // Select output inventories.
  // Default to actor-only. When protocol requires multiple vibes, distribute to participating vibers.
  std::vector<Inventory*> get_output_inventories(const Protocol& protocol,
                                                 const std::vector<Agent*>& surrounding_agents,
                                                 Agent& actor) const {
    if (protocol.vibes.size() <= 1) {
      return {&actor.inventory};
    }

    std::unordered_map<ObservationType, int> required_counts;
    for (ObservationType vibe : protocol.vibes) {
      required_counts[vibe]++;
    }

    std::vector<Inventory*> output_inventories;
    const size_t required = protocol.vibes.size();
    output_inventories.reserve(required);
    for (Agent* agent : surrounding_agents) {
      if (agent->vibe == 0) continue;
      auto it = required_counts.find(agent->vibe);
      if (it == required_counts.end() || it->second <= 0) continue;
      output_inventories.push_back(&agent->inventory);
      it->second--;
      if (output_inventories.size() >= required) break;
    }

    return output_inventories.empty() ? std::vector<Inventory*>{&actor.inventory} : output_inventories;
  }

public:
  // Consume resources from surrounding inventories for the given protocol
  // Intended to be private, but made public for testing. We couldn't get `friend` to work as expected.
  void static consume_resources_for_protocol(const Protocol& protocol,
                                             const std::vector<Inventory*>& surrounding_inventories) {
    for (const auto& [item, required_amount] : protocol.input_resources) {
      InventoryDelta consumed = HasInventory::shared_update(surrounding_inventories, item, -required_amount);
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

  // Game-level stats tracker (shared across environment)
  StatsTracker* stats_tracker;

  // Clipper pointer, for when we become unclipped.
  Clipper* clipper_ptr;

  // Pointer to current timestep from environment
  unsigned int* current_timestep_ptr;

  const class ObservationEncoder* obs_encoder;

  // Allow partial usage during cooldown
  bool allow_partial_usage;

  // Chest search distance - if > 0, assembler can use inventories from chests within this distance
  unsigned int chest_search_distance;

  Assembler(GridCoord r, GridCoord c, const AssemblerConfig& cfg, StatsTracker* stats)
      : protocols(build_protocol_map(cfg.protocols)),
        unclip_protocols(),
        is_clipped(cfg.start_clipped),
        clip_immune(cfg.clip_immune),
        start_clipped(cfg.start_clipped),
        cooldown_end_timestep(0),
        cooldown_duration(0),
        uses_count(0),
        max_uses(cfg.max_uses),
        grid(nullptr),
        stats_tracker(stats),
        current_timestep_ptr(nullptr),
        obs_encoder(nullptr),
        allow_partial_usage(cfg.allow_partial_usage),
        chest_search_distance(cfg.chest_search_distance),
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
    size_t num_agents = get_surrounding_agents(nullptr).size();

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

  // Scale protocol requirements based on cooldown remaining (for partial usage)
  // Uses integer math: elapsed = duration - remaining, then scales by (amount * elapsed) / duration
  const Protocol scale_protocol_for_partial_usage(const Protocol& original_protocol,
                                                  unsigned int elapsed,
                                                  unsigned int duration) const {
    Protocol scaled_protocol;

    // Scale input resources (multiply by elapsed/duration and round up)
    // Round up: (amount * elapsed + duration - 1) / duration
    for (const auto& [resource, amount] : original_protocol.input_resources) {
      InventoryQuantity scaled_amount =
          static_cast<InventoryQuantity>((static_cast<unsigned int>(amount) * elapsed + duration - 1) / duration);
      scaled_protocol.input_resources[resource] = scaled_amount;
    }

    // Scale output resources (multiply by elapsed/duration and round down)
    // Round down: (amount * elapsed) / duration
    for (const auto& [resource, amount] : original_protocol.output_resources) {
      InventoryQuantity scaled_amount =
          static_cast<InventoryQuantity>((static_cast<unsigned int>(amount) * elapsed) / duration);
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
    unsigned int remaining = cooldown_remaining();
    if (remaining > 0 && !allow_partial_usage) {
      return false;  // On cooldown and partial usage not allowed
    }

    const Protocol* original_protocol = get_current_protocol();
    if (!original_protocol) {
      return false;
    }

    Protocol protocol_to_use = *original_protocol;
    if (remaining > 0 && allow_partial_usage) {
      // Calculate elapsed time: duration - remaining
      unsigned int elapsed = cooldown_duration - remaining;
      protocol_to_use = scale_protocol_for_partial_usage(*original_protocol, elapsed, cooldown_duration);

      // Prevent usage that would yield no outputs (and would only serve to burn inputs and increment uses_count)
      // Do not prevent usage if:
      // - the unscaled protocol does not have outputs
      // - usage would unclip the assembler; the unscaled unclipping protocol may happen to include outputs
      if (!Assembler::protocol_has_positive_output(protocol_to_use) &&
          Assembler::protocol_has_positive_output(*original_protocol) && !is_clipped) {
        return false;
      }
    }

    std::vector<Agent*> surrounding_agents = get_surrounding_agents(&actor);
    // Extract Inventory* pointers from agents for resource operations
    std::vector<Inventory*> input_inventories;
    for (Agent* agent : surrounding_agents) {
      input_inventories.push_back(&agent->inventory);
    }
    // Add chest inventories if chest_search_distance > 0
    if (chest_search_distance > 0) {
      std::vector<Chest*> surrounding_chests = get_surrounding_chests();
      for (Chest* chest : surrounding_chests) {
        input_inventories.push_back(&chest->inventory);
      }
    }
    if (!Assembler::can_afford_protocol(protocol_to_use, input_inventories)) {
      return false;
    }
    std::vector<Inventory*> output_inventories = get_output_inventories(protocol_to_use, surrounding_agents, actor);
    if (!Assembler::can_receive_output(protocol_to_use, output_inventories) && !is_clipped) {
      // If the inventories gain nothing from the protocol, don't use it.
      return false;
    }

    consume_resources_for_protocol(protocol_to_use, input_inventories);
    give_output_for_protocol(protocol_to_use, output_inventories);

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
