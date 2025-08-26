// extensions/visitation_counts.cpp
#include "extensions/visitation_counts.hpp"

#include "mettagrid_c.hpp"
#include "objects/agent.hpp"
#include "objects/constants.hpp"
#include "packed_coordinate.hpp"

void VisitationCounts::onInit(const MettaGrid* env) {
  _num_agents = env->num_agents();
  _position_history_r.resize(_num_agents);
  _position_history_c.resize(_num_agents);
}

void VisitationCounts::onReset(MettaGrid* env) {
  for (size_t i = 0; i < _num_agents; i++) {
    _position_history_r[i].clear();
    _position_history_c[i].clear();
    _position_history_r[i].reserve(env->max_steps);
    _position_history_c[i].reserve(env->max_steps);

    const Agent* agent = env->agent(static_cast<uint32_t>(i));
    _position_history_r[i].push_back(agent->location.r);
    _position_history_c[i].push_back(agent->location.c);
  }
}

void VisitationCounts::onStep(MettaGrid* env) {
  // Record current positions
  for (size_t i = 0; i < _num_agents; i++) {
    const Agent* agent = env->agent(static_cast<uint32_t>(i));
    _position_history_r[i].push_back(agent->location.r);
    _position_history_c[i].push_back(agent->location.c);
  }

  // Inject visitation counts into observations
  auto& observations = getObservations(env);  // Use protected accessor
  auto obs_view = observations.mutable_unchecked<3>();

  for (size_t agent_idx = 0; agent_idx < _num_agents; agent_idx++) {
    auto counts = computeVisitationCounts(env, agent_idx);

    uint8_t center_r = env->obs_height / 2;
    uint8_t center_c = env->obs_width / 2;
    uint8_t center_packed = PackedCoordinate::pack(center_r, center_c);

    size_t insert_pos = 0;
    for (ssize_t token_idx = 0; token_idx < obs_view.shape(1); token_idx++) {
      if (obs_view(agent_idx, token_idx, 0) == center_packed) {
        while (token_idx < obs_view.shape(1) && obs_view(agent_idx, token_idx, 0) == center_packed) {
          token_idx++;
        }
        insert_pos = static_cast<size_t>(token_idx);
        break;
      }
    }

    if (static_cast<ssize_t>(insert_pos + 5) <= obs_view.shape(1)) {
      for (size_t i = 0; i < 5; i++) {
        obs_view(agent_idx, insert_pos + i, 0) = center_packed;
        obs_view(agent_idx, insert_pos + i, 1) = ObservationFeature::VisitationCounts;
        obs_view(agent_idx, insert_pos + i, 2) = static_cast<uint8_t>(std::min(counts[i], 255u));
      }
    }
  }
}

py::dict VisitationCounts::getStats() const {
  py::dict stats;
  py::list agent_stats;

  for (size_t i = 0; i < _num_agents; i++) {
    py::dict agent_stat;
    agent_stat["total_steps"] = _position_history_r[i].size();

    std::set<std::pair<GridCoord, GridCoord>> unique_positions;
    for (size_t j = 0; j < _position_history_r[i].size(); j++) {
      unique_positions.insert({_position_history_r[i][j], _position_history_c[i][j]});
    }
    agent_stat["unique_cells_visited"] = unique_positions.size();

    agent_stats.append(agent_stat);
  }

  stats["agents"] = agent_stats;
  return stats;
}

std::array<unsigned int, 5> VisitationCounts::computeVisitationCounts(const MettaGrid* env, size_t agent_idx) const {
  const Agent* agent = env->agent(static_cast<uint32_t>(agent_idx));
  std::array<unsigned int, 5> counts = {0, 0, 0, 0, 0};

  GridCoord current_r = agent->location.r;
  GridCoord current_c = agent->location.c;

  for (size_t i = 0; i < _position_history_r[agent_idx].size(); i++) {
    GridCoord hist_r = _position_history_r[agent_idx][i];
    GridCoord hist_c = _position_history_c[agent_idx][i];

    if (hist_r == current_r && hist_c == current_c) {
      counts[0]++;
    } else if (hist_r == current_r - 1 && hist_c == current_c) {
      counts[1]++;
    } else if (hist_r == current_r + 1 && hist_c == current_c) {
      counts[2]++;
    } else if (hist_r == current_r && hist_c == current_c - 1) {
      counts[3]++;
    } else if (hist_r == current_r && hist_c == current_c + 1) {
      counts[4]++;
    }
  }

  return counts;
}

REGISTER_EXTENSION("visitation_counts", VisitationCounts)
