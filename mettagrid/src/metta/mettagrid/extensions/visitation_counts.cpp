// extensions/visitation_counts.cpp
#include "extensions/visitation_counts.hpp"

#include "mettagrid_c.hpp"
#include "objects/agent.hpp"
#include "packed_coordinate.hpp"

void VisitationCounts::registerObservations(ObservationEncoder* enc) {
  _visitation_count_feature = enc->register_feature("agent:visitation_counts");
}

void VisitationCounts::onInit(const MettaGrid* env) {
  _num_agents = env->num_agents();
  _position_history_r.resize(_num_agents);
  _position_history_c.resize(_num_agents);
}

void VisitationCounts::onReset(MettaGrid* env) {
  // Initialize position history
  for (size_t i = 0; i < _num_agents; i++) {
    _position_history_r[i].clear();
    _position_history_c[i].clear();
    _position_history_r[i].reserve(env->max_steps);
    _position_history_c[i].reserve(env->max_steps);

    const Agent* agent = env->agent(static_cast<uint32_t>(i));
    _position_history_r[i].push_back(agent->location.r);
    _position_history_c[i].push_back(agent->location.c);
  }

  // Add initial visitation counts to observations
  addVisitationCountsToObservations(env);
}

void VisitationCounts::onStep(MettaGrid* env) {
  // Record current positions
  for (size_t i = 0; i < _num_agents; i++) {
    const Agent* agent = env->agent(static_cast<uint32_t>(i));
    _position_history_r[i].push_back(agent->location.r);
    _position_history_c[i].push_back(agent->location.c);
  }

  // Add visitation counts to observations
  addVisitationCountsToObservations(env);
}

std::array<unsigned int, 5> VisitationCounts::computeVisitationCounts(const MettaGrid* env, size_t agent_idx) const {
  // Count visits to the current cell and 4 adjacent cells (N, S, E, W)
  std::array<unsigned int, 5> counts = {0, 0, 0, 0, 0};

  if (agent_idx >= _num_agents || _position_history_r[agent_idx].empty()) {
    return counts;
  }

  // Get current agent position
  const Agent* agent = env->agent(static_cast<uint32_t>(agent_idx));
  GridCoord curr_r = agent->location.r;
  GridCoord curr_c = agent->location.c;

  // Count visits to each position
  for (size_t i = 0; i < _position_history_r[agent_idx].size(); i++) {
    GridCoord hist_r = _position_history_r[agent_idx][i];
    GridCoord hist_c = _position_history_c[agent_idx][i];

    // Check center position
    if (hist_r == curr_r && hist_c == curr_c) {
      counts[0]++;
    }

    // Check north (r-1)
    if (curr_r > 0 && hist_r == curr_r - 1 && hist_c == curr_c) {
      counts[1]++;
    }

    // Check south (r+1)
    if (hist_r == curr_r + 1 && hist_c == curr_c) {
      counts[2]++;
    }

    // Check east (c+1)
    if (hist_r == curr_r && hist_c == curr_c + 1) {
      counts[3]++;
    }

    // Check west (c-1)
    if (curr_c > 0 && hist_r == curr_r && hist_c == curr_c - 1) {
      counts[4]++;
    }
  }

  return counts;
}

void VisitationCounts::addVisitationCountsToObservations(MettaGrid* env) {
  // Get dimensions
  size_t num_tokens = env->num_observation_tokens;
  size_t num_channels = 3;  // packed_coord, feature, value

  for (size_t agent_idx = 0; agent_idx < _num_agents; agent_idx++) {
    auto agent_obs = getAgentObservationsMutable(env, agent_idx);
    auto counts = computeVisitationCounts(env, agent_idx);

    uint8_t center_r = env->obs_height / 2;
    uint8_t center_c = env->obs_width / 2;

    // Define positions for the 5 counts (center + 4 adjacent)
    std::array<std::pair<uint8_t, uint8_t>, 5> positions;
    positions[0] = {center_r, center_c};  // center

    // North (r-1) - check for underflow
    positions[1] = {(center_r > 0) ? static_cast<uint8_t>(center_r - 1) : uint8_t(0xFF), center_c};

    // South (r+1)
    positions[2] = {static_cast<uint8_t>(center_r + 1), center_c};

    // East (c+1)
    positions[3] = {center_r, static_cast<uint8_t>(center_c + 1)};

    // West (c-1) - check for underflow
    positions[4] = {center_r, (center_c > 0) ? static_cast<uint8_t>(center_c - 1) : uint8_t(0xFF)};

    // Helper to access the 2D observation array for this agent
    auto get_obs = [&](size_t token_idx, size_t channel) -> uint8_t& {
      return agent_obs[token_idx * num_channels + channel];
    };

    // Find the first completely empty slot
    size_t insert_pos = 0;
    bool found_empty = false;
    for (size_t token_idx = 0; token_idx < num_tokens; token_idx++) {
      if (get_obs(token_idx, 0) == 0xFF && get_obs(token_idx, 1) == 0xFF && get_obs(token_idx, 2) == 0xFF) {
        insert_pos = token_idx;
        found_empty = true;
        break;
      }
    }

    // Insert as many tokens as will fit
    if (found_empty) {
      size_t tokens_to_insert = std::min(static_cast<size_t>(5), num_tokens - insert_pos);
      for (size_t i = 0; i < tokens_to_insert; i++) {
        // Skip positions that are marked as invalid (0xFF)
        if (positions[i].first != 0xFF && positions[i].second != 0xFF && positions[i].first < env->obs_height &&
            positions[i].second < env->obs_width) {
          uint8_t packed_loc = PackedCoordinate::pack(positions[i].first, positions[i].second);
          get_obs(insert_pos + i, 0) = packed_loc;
          get_obs(insert_pos + i, 1) = static_cast<uint8_t>(_visitation_count_feature);
          get_obs(insert_pos + i, 2) = static_cast<uint8_t>(std::min(counts[i], 255u));
        }
      }
    }
  }
}

ExtensionStats VisitationCounts::getStats() const {
  ExtensionStats stats;

  // Calculate average stats across all agents
  float avg_total_steps = 0.0f;
  float avg_unique_cells = 0.0f;

  for (size_t i = 0; i < _num_agents; i++) {
    avg_total_steps += static_cast<float>(_position_history_r[i].size());

    std::set<std::pair<GridCoord, GridCoord>> unique_positions;
    for (size_t j = 0; j < _position_history_r[i].size(); j++) {
      unique_positions.insert({_position_history_r[i][j], _position_history_c[i][j]});
    }
    avg_unique_cells += static_cast<float>(unique_positions.size());
  }

  if (_num_agents > 0) {
    avg_total_steps /= static_cast<float>(_num_agents);
    avg_unique_cells /= static_cast<float>(_num_agents);
  }

  stats["avg_agent_total_steps"] = avg_total_steps;
  stats["avg_agent_unique_cells_visited"] = avg_unique_cells;

  return stats;
}

REGISTER_EXTENSION("visitation_counts", VisitationCounts)
