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
  uint8_t center_r = env->obs_height / 2;
  uint8_t center_c = env->obs_width / 2;

  for (size_t agent_idx = 0; agent_idx < _num_agents; agent_idx++) {
    auto counts = computeVisitationCounts(env, agent_idx);

    // Create observation tokens for the 5 visitation counts
    std::vector<ObservationToken> tokens;

    // Center position
    ObservationToken center_token;
    center_token.location = PackedCoordinate::pack(center_r, center_c);
    center_token.feature_id = _visitation_count_feature;
    center_token.value = static_cast<uint8_t>(std::min(counts[0], 255u));
    tokens.push_back(center_token);

    // North (r-1) - check bounds
    if (center_r > 0) {
      ObservationToken north_token;
      north_token.location = PackedCoordinate::pack(center_r - 1, center_c);
      north_token.feature_id = _visitation_count_feature;
      north_token.value = static_cast<uint8_t>(std::min(counts[1], 255u));
      tokens.push_back(north_token);
    }

    // South (r+1) - check bounds
    if (center_r + 1 < env->obs_height) {
      ObservationToken south_token;
      south_token.location = PackedCoordinate::pack(center_r + 1, center_c);
      south_token.feature_id = _visitation_count_feature;
      south_token.value = static_cast<uint8_t>(std::min(counts[2], 255u));
      tokens.push_back(south_token);
    }

    // East (c+1) - check bounds
    if (center_c + 1 < env->obs_width) {
      ObservationToken east_token;
      east_token.location = PackedCoordinate::pack(center_r, center_c + 1);
      east_token.feature_id = _visitation_count_feature;
      east_token.value = static_cast<uint8_t>(std::min(counts[3], 255u));
      tokens.push_back(east_token);
    }

    // West (c-1) - check bounds
    if (center_c > 0) {
      ObservationToken west_token;
      west_token.location = PackedCoordinate::pack(center_r, center_c - 1);
      west_token.feature_id = _visitation_count_feature;
      west_token.value = static_cast<uint8_t>(std::min(counts[4], 255u));
      tokens.push_back(west_token);
    }

    // Write all tokens at once
    writeObservations(env, agent_idx, tokens);
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
