#ifndef EXTENSIONS_DELTA_MOTION_HPP_
#define EXTENSIONS_DELTA_MOTION_HPP_

#include <algorithm>
#include <cstdint>
#include <deque>
#include <vector>

#include "extensions/mettagrid_extension.hpp"
#include "mettagrid_c.hpp"
#include "objects/agent.hpp"
#include "packed_coordinate.hpp"
#include "types.hpp"

// Extension that tracks integrated motion (cumulative dx, dy) over a rolling window
// of the last 15 steps for each agent. Provides this as a global observation token.

class DeltaMotion : public MettaGridExtension {
public:
  void registerObservations(ObservationEncoder* enc) override {
    _delta_motion_feature = enc->register_feature("agent:delta_motion");
  }

  void onInit(const MettaGrid* env) override {
    _num_agents = env->num_agents();

    // Pre-allocate storage
    _prev_positions.resize(_num_agents);
    _motion_history.resize(_num_agents);
  }

  void onReset(MettaGrid* env) override {
    // Clear motion history and set initial positions
    for (size_t i = 0; i < _num_agents; i++) {
      const Agent* agent = getAgent(env, i);
      _prev_positions[i] = {agent->location.r, agent->location.c};
      _motion_history[i].clear();
    }

    // Write initial observations (zero motion)
    uint8_t initial_packed = PackedCoordinate::pack(ZERO_MOTION_PACKED, ZERO_MOTION_PACKED);

    for (size_t i = 0; i < _num_agents; i++) {
      writeGlobalObservations(env, i, {_delta_motion_feature}, {initial_packed});
    }
  }

  void onStep(MettaGrid* env) override {
    for (size_t i = 0; i < _num_agents; i++) {
      const Agent* agent = getAgent(env, i);

      // Calculate motion delta
      Motion delta;
      delta.dx = static_cast<int8_t>(agent->location.c - _prev_positions[i].c);
      delta.dy = static_cast<int8_t>(agent->location.r - _prev_positions[i].r);

      // Update motion history
      _motion_history[i].push_back(delta);
      if (_motion_history[i].size() > WINDOW_SIZE) {
        _motion_history[i].pop_front();
      }

      // Update previous position
      _prev_positions[i] = {agent->location.r, agent->location.c};

      // Compute integrated motion
      Motion integrated = computeIntegratedMotion(i);
      uint8_t packed_motion = packIntegratedMotion(integrated);

      // Write observation
      writeGlobalObservations(env, i, {_delta_motion_feature}, {packed_motion});
    }
  }

  std::string getName() const override {
    return "delta_motion";
  }

private:
  static constexpr size_t WINDOW_SIZE = 15;

  // Constants for motion value mapping
  // We map integrated motion from [-WINDOW_SIZE, +WINDOW_SIZE] to [0, MAX_PACKABLE_COORD]
  static constexpr uint8_t MOTION_OFFSET = WINDOW_SIZE;  // Offset to make motion values positive
  static constexpr uint8_t MOTION_SCALE = 2;             // Scale factor to fit in PackedCoordinate range
  static constexpr uint8_t ZERO_MOTION_PACKED = PackedCoordinate::MAX_PACKABLE_COORD / 2;  // Center of range

  struct Motion {
    int8_t dx;  // Change in column (east/west)
    int8_t dy;  // Change in row (north/south)
  };

  struct Position {
    GridCoord r;
    GridCoord c;
  };

  size_t _num_agents;
  ObservationType _delta_motion_feature;

  // Previous positions for each agent
  std::vector<Position> _prev_positions;

  // Rolling window of last 15 motions for each agent
  std::vector<std::deque<Motion>> _motion_history;

  // Helper to compute integrated motion
  Motion computeIntegratedMotion(size_t agent_idx) const {
    Motion integrated = {0, 0};

    for (const auto& motion : _motion_history[agent_idx]) {
      integrated.dx += motion.dx;
      integrated.dy += motion.dy;
    }

    return integrated;
  }

  // Helper to pack integrated motion into uint8_t using PackedCoordinate
  // Maps integrated motion values from [-15,15] to packed coordinates [0,14]
  uint8_t packIntegratedMotion(const Motion& motion) const {
    // Map from [-15, 15] to [0, 30] then divide by 2 to fit in [0, 14]
    int mapped_dx = (motion.dx + MOTION_OFFSET) / MOTION_SCALE;
    int mapped_dy = (motion.dy + MOTION_OFFSET) / MOTION_SCALE;

    // Clamp to valid PackedCoordinate range [0, MAX_PACKABLE_COORD]
    uint8_t coord_x =
        static_cast<uint8_t>(std::max(0, std::min(static_cast<int>(PackedCoordinate::MAX_PACKABLE_COORD), mapped_dx)));
    uint8_t coord_y =
        static_cast<uint8_t>(std::max(0, std::min(static_cast<int>(PackedCoordinate::MAX_PACKABLE_COORD), mapped_dy)));

    return PackedCoordinate::pack(coord_y, coord_x);
  }
};

REGISTER_EXTENSION("delta_motion", DeltaMotion)

#endif  // EXTENSIONS_DELTA_MOTION_HPP_
