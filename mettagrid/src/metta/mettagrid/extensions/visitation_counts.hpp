// extensions/visitation_counts.hpp
#pragma once

#include <array>
#include <unordered_map>
#include <vector>

#include "extension.hpp"

class VisitationCounts : public Extension {
public:
  void registerObservations(ObservationEncoder* enc) override;
  void onInit(const MettaGrid* env) override;
  void onReset(MettaGrid* env) override;
  void onStep(MettaGrid* env) override;

  ExtensionStats getStats() const override;

private:
  size_t _num_agents;
  size_t _grid_width;
  size_t _grid_height;
  bool _use_dense;

  // Dense storage for small grids
  std::vector<uint8_t> _dense_visits;

  // Sparse storage for large grids
  std::vector<std::unordered_map<uint32_t, uint8_t>> _sparse_visits;

  ObservationFeatureID _visitation_count_feature;

  // Helper to get index in dense array
  inline size_t getDenseIndex(size_t agent_idx, GridCoord r, GridCoord c) const {
    return agent_idx * _grid_width * _grid_height + r * _grid_width + c;
  }

  std::array<unsigned int, 5> computeVisitationCounts(const MettaGrid* env, size_t agent_idx) const;
  void addVisitationCountsToObservations(MettaGrid* env);
};
