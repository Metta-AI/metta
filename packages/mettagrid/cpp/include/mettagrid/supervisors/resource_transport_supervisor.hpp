#ifndef PACKAGES_METTAGRID_CPP_INCLUDE_METTAGRID_SUPERVISORS_RESOURCE_TRANSPORT_SUPERVISOR_HPP_
#define PACKAGES_METTAGRID_CPP_INCLUDE_METTAGRID_SUPERVISORS_RESOURCE_TRANSPORT_SUPERVISOR_HPP_

#include <algorithm>
#include <limits>
#include <optional>
#include <unordered_map>
#include <vector>

#include "core/grid.hpp"
#include "core/grid_object.hpp"
#include "objects/agent.hpp"
#include "objects/chest.hpp"
#include "objects/converter.hpp"
#include "objects/wall.hpp"
#include "supervisors/agent_supervisor.hpp"
#include "systems/packed_coordinate.hpp"

// Configuration for ResourceTransportSupervisor
struct ResourceTransportSupervisorConfig : public AgentSupervisorConfig {
  // The resource type to transport (e.g., "Carbon")
  InventoryItem target_resource;
  // Minimum energy level to maintain before seeking energy
  InventoryQuantity min_energy_threshold;
  // Whether to manage energy (seek energy sources when low)
  bool manage_energy;
  // Maximum distance to search for objects
  GridCoord max_search_distance;
  // Grid dimensions for obstacle tracking
  GridCoord grid_width;
  GridCoord grid_height;
  // Observation window dimensions
  ObservationCoord obs_width;
  ObservationCoord obs_height;

  ResourceTransportSupervisorConfig(InventoryItem target_resource,
                                    InventoryQuantity min_energy_threshold = 10,
                                    bool manage_energy = true,
                                    GridCoord max_search_distance = 30,
                                    bool can_override_action = false,
                                    const std::string& name = "resource_transport_supervisor",
                                    GridCoord grid_width = 100,
                                    GridCoord grid_height = 100,
                                    ObservationCoord obs_width = 11,
                                    ObservationCoord obs_height = 11)
      : AgentSupervisorConfig(can_override_action, name),
        target_resource(target_resource),
        min_energy_threshold(min_energy_threshold),
        manage_energy(manage_energy),
        max_search_distance(max_search_distance),
        grid_width(grid_width),
        grid_height(grid_height),
        obs_width(obs_width),
        obs_height(obs_height) {}
};

// Supervisor that manages resource transportation from extractors to chests
class ResourceTransportSupervisor : public AgentSupervisor {
public:
  explicit ResourceTransportSupervisor(const ResourceTransportSupervisorConfig& config)
      : AgentSupervisor(config),
        config_(config),
        state_(State::SEARCHING),
        target_extractor_(nullptr),
        target_chest_(nullptr),
        search_index_(0),
        obstacle_map_(config.grid_height, std::vector<bool>(config.grid_width, false)) {}

  void reset() override {
    state_ = State::SEARCHING;
    target_extractor_ = nullptr;
    target_chest_ = nullptr;
    search_index_ = 0;
    visited_locations_.clear();
    // Reset obstacle map
    for (auto& row : obstacle_map_) {
      std::fill(row.begin(), row.end(), false);
    }
  }

  // Get a read-only view of the obstacle map (for debugging/visualization)
  const std::vector<std::vector<bool>>& get_obstacle_map() const {
    return obstacle_map_;
  }

  // Check if a specific position is marked as an obstacle
  bool is_obstacle_at(GridCoord r, GridCoord c) const {
    if (r >= 0 && r < static_cast<GridCoord>(config_.grid_height) && c >= 0 &&
        c < static_cast<GridCoord>(config_.grid_width)) {
      return obstacle_map_[r][c];
    }
    return false;
  }

protected:
  std::pair<ActionType, ActionArg> get_recommended_action(const ObservationTokens& observation) override {
    // Update obstacle map based on observations
    update_obstacle_map(observation);

    // Check energy levels if energy management is enabled
    if (config_.manage_energy && needs_energy()) {
      return seek_energy();
    }

    // State machine for resource transportation
    switch (state_) {
      case State::SEARCHING:
        return search_for_targets();

      case State::MOVING_TO_EXTRACTOR:
        return move_to_extractor();

      case State::EXTRACTING:
        return extract_resource();

      case State::MOVING_TO_CHEST:
        return move_to_chest();

      case State::DEPOSITING:
        return deposit_resource();

      default:
        // Fallback to no-op
        return {0, 0};  // Assuming action 0 is noop
    }
  }

  void on_supervise(ActionType agent_action,
                    ActionArg agent_arg,
                    ActionType supervisor_action,
                    ActionArg supervisor_arg,
                    bool agrees) override {
    // Record state-specific statistics
    std::string state_name = get_state_name();
    agent_->stats.incr(name_ + ".state." + state_name);

    if (!agrees) {
      agent_->stats.incr(name_ + ".state." + state_name + ".disagree");
    }
  }

private:
  enum class State {
    SEARCHING,
    MOVING_TO_EXTRACTOR,
    EXTRACTING,
    MOVING_TO_CHEST,
    DEPOSITING
  };

  ResourceTransportSupervisorConfig config_;
  State state_;
  Converter* target_extractor_;
  Chest* target_chest_;
  size_t search_index_;
  std::vector<GridLocation> visited_locations_;
  std::vector<std::vector<bool>> obstacle_map_;

  void update_obstacle_map(const ObservationTokens& observation) {
    // Calculate observation window radius
    ObservationCoord obs_height_radius = config_.obs_height >> 1;
    ObservationCoord obs_width_radius = config_.obs_width >> 1;

    // Process each observation token
    for (const auto& token : observation) {
      // Skip empty tokens
      if (PackedCoordinate::is_empty(token.location)) {
        continue;
      }

      // Unpack the location within the observation window
      auto unpacked = PackedCoordinate::unpack(token.location);
      if (!unpacked.has_value()) {
        continue;
      }

      auto [obs_r, obs_c] = unpacked.value();

      // The observation window is centered on the agent
      // Calculate the actual grid position from the relative observation position
      int grid_r = agent_->location.r + static_cast<int>(obs_r) - static_cast<int>(obs_height_radius);
      int grid_c = agent_->location.c + static_cast<int>(obs_c) - static_cast<int>(obs_width_radius);

      // Check if the position is within grid bounds
      if (grid_r < 0 || grid_r >= static_cast<int>(config_.grid_height) || grid_c < 0 ||
          grid_c >= static_cast<int>(config_.grid_width)) {
        continue;
      }

      // Check if this token represents an object (not empty)
      // We need to check if it's not an agent to mark it as an obstacle
      // The feature_id would tell us what type of object it is
      // For simplicity, we'll mark any non-empty, non-agent object as an obstacle

      // Check if there's an object at this location in the grid
      GridLocation loc(grid_r, grid_c, GridLayer::ObjectLayer);
      GridObject* obj = grid_->object_at(loc);
      if (obj != nullptr) {
        // Check if it's not an agent
        Agent* agent = dynamic_cast<Agent*>(obj);
        if (agent == nullptr) {
          // It's not an agent, so mark it as an obstacle
          obstacle_map_[grid_r][grid_c] = true;
        }
      }
    }
  }

  bool needs_energy() const {
    // Check if agent has an energy resource and it's below threshold
    // This assumes energy is resource 0 or we need to determine the energy resource ID
    // For now, we'll check if any resource that could be "energy" is low
    // This is a simplified check - in practice you'd want to configure the energy resource
    return false;  // Placeholder - implement based on game configuration
  }

  std::pair<ActionType, ActionArg> seek_energy() {
    // Find and move to nearest energy source
    // This is a placeholder - implement based on game configuration
    return {0, 0};  // Return noop for now
  }

  std::pair<ActionType, ActionArg> search_for_targets() {
    // Find extractor and chest for the target resource
    if (!target_extractor_ || !target_chest_) {
      find_resource_infrastructure();
    }

    if (target_extractor_ && target_chest_) {
      state_ = State::MOVING_TO_EXTRACTOR;
      return move_to_extractor();
    }

    // Explore the map systematically
    return explore_map();
  }

  void find_resource_infrastructure() {
    GridCoord agent_r = agent_->location.r;
    GridCoord agent_c = agent_->location.c;

    // Search for converters that produce the target resource
    for (GridCoord r = 0; r < grid_->height; ++r) {
      for (GridCoord c = 0; c < grid_->width; ++c) {
        // Check distance constraint
        GridCoord dist = std::abs(r - agent_r) + std::abs(c - agent_c);
        if (dist > config_.max_search_distance) {
          continue;
        }

        // Check for converter at this location
        GridLocation loc(r, c, GridLayer::ObjectLayer);
        GridObject* obj = grid_->object_at(loc);
        if (!obj) continue;

        Converter* converter = dynamic_cast<Converter*>(obj);
        if (converter && produces_resource(converter, config_.target_resource)) {
          if (!target_extractor_ || distance_to(converter) < distance_to(target_extractor_)) {
            target_extractor_ = converter;
          }
        }

        Chest* chest = dynamic_cast<Chest*>(obj);
        if (chest && can_store_resource(chest, config_.target_resource)) {
          if (!target_chest_ || distance_to(chest) < distance_to(target_chest_)) {
            target_chest_ = chest;
          }
        }
      }
    }
  }

  bool produces_resource(Converter* converter, InventoryItem resource) const {
    // Check if converter produces the target resource
    // For now, we'll assume any converter can produce the resource
    // In a full implementation, we'd check the converter's configuration
    return true;  // Simplified check
  }

  bool can_store_resource(Chest* chest, InventoryItem resource) const {
    // Check if chest can store the resource (has capacity)
    // For now assume all chests can store resources if not full
    return chest->inventory.amount(resource) < 1000;  // Simplified check
  }

  GridCoord distance_to(GridObject* obj) const {
    return std::abs(agent_->location.r - obj->location.r) + std::abs(agent_->location.c - obj->location.c);
  }

  std::pair<ActionType, ActionArg> move_to_extractor() {
    if (!target_extractor_) {
      state_ = State::SEARCHING;
      return search_for_targets();
    }

    // Check if we're adjacent to the extractor
    if (is_adjacent(target_extractor_)) {
      state_ = State::EXTRACTING;
      return extract_resource();
    }

    return move_toward(target_extractor_->location);
  }

  std::pair<ActionType, ActionArg> extract_resource() {
    if (!target_extractor_) {
      state_ = State::SEARCHING;
      return search_for_targets();
    }

    // Check if extractor has resources available
    InventoryQuantity available = target_extractor_->inventory.amount(config_.target_resource);
    if (available == 0) {
      // Extractor is empty, search for a new one
      target_extractor_ = nullptr;
      state_ = State::SEARCHING;
      return search_for_targets();
    }

    // Check if agent inventory is full
    InventoryQuantity agent_amount = agent_->inventory.amount(config_.target_resource);
    InventoryQuantity agent_limit = 100;  // Simplified limit check

    if (agent_amount >= agent_limit) {
      // Agent is full, move to chest
      state_ = State::MOVING_TO_CHEST;
      return move_to_chest();
    }

    // Extract resources (assuming action 3 is get_items)
    // The arg would be the direction to the extractor
    ActionArg direction = get_direction_to(target_extractor_);
    return {3, direction};  // Action 3 = get_items
  }

  std::pair<ActionType, ActionArg> move_to_chest() {
    if (!target_chest_) {
      state_ = State::SEARCHING;
      return search_for_targets();
    }

    // Check if we're adjacent to the chest
    if (is_adjacent(target_chest_)) {
      state_ = State::DEPOSITING;
      return deposit_resource();
    }

    return move_toward(target_chest_->location);
  }

  std::pair<ActionType, ActionArg> deposit_resource() {
    if (!target_chest_) {
      state_ = State::SEARCHING;
      return search_for_targets();
    }

    // Check if agent has resources to deposit
    InventoryQuantity agent_amount = agent_->inventory.amount(config_.target_resource);
    if (agent_amount == 0) {
      // No resources to deposit, go back to extractor
      state_ = State::MOVING_TO_EXTRACTOR;
      return move_to_extractor();
    }

    // Check if chest is full
    if (!can_store_resource(target_chest_, config_.target_resource)) {
      // Chest is full, find a new one
      target_chest_ = nullptr;
      state_ = State::SEARCHING;
      return search_for_targets();
    }

    // Deposit resources (assuming action 2 is put_items)
    ActionArg direction = get_direction_to(target_chest_);
    return {2, direction};  // Action 2 = put_items
  }

  std::pair<ActionType, ActionArg> explore_map() {
    // Systematic exploration pattern
    // Move in a spiral or grid pattern to discover the map

    // For now, just move randomly to unexplored areas
    std::vector<ActionArg> possible_moves;

    // Check each direction
    for (ActionArg dir = 0; dir < 4; ++dir) {
      GridLocation next_loc = get_location_in_direction(agent_->location, dir);

      // Check if location is valid and not recently visited
      if (is_valid_location(next_loc) && !is_recently_visited(next_loc)) {
        possible_moves.push_back(dir);
      }
    }

    if (!possible_moves.empty()) {
      // Choose the least recently visited direction
      ActionArg chosen_dir = possible_moves[0];
      GridLocation chosen_loc = get_location_in_direction(agent_->location, chosen_dir);
      visited_locations_.push_back(chosen_loc);

      // Keep visited locations list bounded
      if (visited_locations_.size() > 100) {
        visited_locations_.erase(visited_locations_.begin());
      }

      return {1, chosen_dir};  // Action 1 = move
    }

    // No good moves, just wait
    return {0, 0};  // noop
  }

  std::pair<ActionType, ActionArg> move_toward(const GridLocation& target) {
    // Calculate best direction to move toward target
    GridCoord dr = target.r - agent_->location.r;
    GridCoord dc = target.c - agent_->location.c;

    // Prefer moving in the dimension with greater distance
    ActionArg direction;
    if (std::abs(dr) > std::abs(dc)) {
      direction = (dr > 0) ? 2 : 0;  // South : North
    } else if (dc != 0) {
      direction = (dc > 0) ? 1 : 3;  // East : West
    } else if (dr != 0) {
      direction = (dr > 0) ? 2 : 0;  // South : North
    } else {
      return {0, 0};  // Already at target
    }

    // Check if the move is valid
    GridLocation next_loc = get_location_in_direction(agent_->location, direction);
    if (is_valid_location(next_loc)) {
      return {1, direction};  // Action 1 = move
    }

    // Try alternative directions
    for (ActionArg alt_dir = 0; alt_dir < 4; ++alt_dir) {
      if (alt_dir == direction) continue;

      GridLocation alt_loc = get_location_in_direction(agent_->location, alt_dir);
      if (is_valid_location(alt_loc)) {
        return {1, alt_dir};
      }
    }

    return {0, 0};  // No valid moves
  }

  bool is_adjacent(GridObject* obj) const {
    GridCoord dist = std::abs(agent_->location.r - obj->location.r) + std::abs(agent_->location.c - obj->location.c);
    return dist == 1;
  }

  ActionArg get_direction_to(GridObject* obj) const {
    GridCoord dr = obj->location.r - agent_->location.r;
    GridCoord dc = obj->location.c - agent_->location.c;

    if (dr == -1 && dc == 0) return 0;  // North
    if (dr == 0 && dc == 1) return 1;   // East
    if (dr == 1 && dc == 0) return 2;   // South
    if (dr == 0 && dc == -1) return 3;  // West

    return 0;  // Default
  }

  GridLocation get_location_in_direction(const GridLocation& loc, ActionArg direction) const {
    GridLocation new_loc = loc;
    switch (direction) {
      case 0:
        new_loc.r -= 1;
        break;  // North
      case 1:
        new_loc.c += 1;
        break;  // East
      case 2:
        new_loc.r += 1;
        break;  // South
      case 3:
        new_loc.c -= 1;
        break;  // West
    }
    return new_loc;
  }

  bool is_valid_location(const GridLocation& loc) const {
    // Check bounds against actual grid dimensions
    if (loc.r < 0 || loc.r >= static_cast<GridCoord>(config_.grid_height) || loc.c < 0 ||
        loc.c >= static_cast<GridCoord>(config_.grid_width)) {
      return false;
    }

    // Check if location is marked as an obstacle in our map
    if (obstacle_map_[loc.r][loc.c]) {
      return false;
    }

    // Check if location is blocked
    GridObject* obj = grid_->object_at(loc);
    if (obj) {
      // Check if it's a wall or other blocking object
      Wall* wall = dynamic_cast<Wall*>(obj);
      if (wall) return false;
    }

    return true;
  }

  bool is_recently_visited(const GridLocation& loc) const {
    // Check if location was visited recently (in last N moves)
    size_t recent_count = std::min(size_t(10), visited_locations_.size());
    for (size_t i = visited_locations_.size() - recent_count; i < visited_locations_.size(); ++i) {
      if (visited_locations_[i] == loc) {
        return true;
      }
    }
    return false;
  }

  std::string get_state_name() const {
    switch (state_) {
      case State::SEARCHING:
        return "searching";
      case State::MOVING_TO_EXTRACTOR:
        return "moving_to_extractor";
      case State::EXTRACTING:
        return "extracting";
      case State::MOVING_TO_CHEST:
        return "moving_to_chest";
      case State::DEPOSITING:
        return "depositing";
      default:
        return "unknown";
    }
  }
};

#endif  // PACKAGES_METTAGRID_CPP_INCLUDE_METTAGRID_SUPERVISORS_RESOURCE_TRANSPORT_SUPERVISOR_HPP_
