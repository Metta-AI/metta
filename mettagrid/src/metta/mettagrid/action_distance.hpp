#ifndef ACTION_DISTANCE_HPP_
#define ACTION_DISTANCE_HPP_

#include <algorithm>
#include <array>
#include <cstdint>
#include <memory>
#include <string>
#include <vector>

#include "types.hpp"  // For ActionType, ActionArg

namespace ActionDistance {

// Semantic action groups for distance calculation
enum ActionGroup : uint8_t {
  GROUP_IDLE = 0,
  GROUP_INVENTORY = 1,
  GROUP_SPATIAL = 2,
  GROUP_COMBAT = 3,
  GROUP_DISPLAY = 4,
  NUM_GROUPS = 5
};

// Base distances between semantic groups
constexpr uint8_t GROUP_DISTANCES[NUM_GROUPS][NUM_GROUPS] = {
    // IDLE  INV   SPAT  COMB  DISP
    {0, 10, 10, 10, 10},  // IDLE
    {10, 0, 4, 6, 8},     // INVENTORY
    {10, 4, 0, 3, 6},     // SPATIAL
    {10, 6, 3, 0, 7},     // COMBAT
    {10, 8, 6, 7, 0}      // DISPLAY
};

// Map action names to semantic groups
inline ActionGroup get_action_group(const std::string& action_name) {
  if (action_name == "noop") return GROUP_IDLE;
  if (action_name == "put_items" || action_name == "get_items") return GROUP_INVENTORY;
  if (action_name == "move" || action_name == "rotate" || action_name == "swap") return GROUP_SPATIAL;
  if (action_name == "attack") return GROUP_COMBAT;
  if (action_name == "change_glyph" || action_name == "change_color") return GROUP_DISPLAY;
  return GROUP_IDLE;  // Default to idle for unknown actions
}

// Special case handlers for intra-action distances
inline uint8_t calculate_move_distance(uint8_t arg1, uint8_t arg2) {
  // Forward(0) vs Back(1)
  return (arg1 != arg2) ? 3 : 0;  // Opposites have higher distance
}

inline uint8_t calculate_rotate_distance(uint8_t arg1, uint8_t arg2) {
  // Up=0, Down=1, Left=2, Right=3
  // Adjacent rotations (Up-Left, Left-Down, etc.) = 1
  // Opposite rotations (Up-Down, Left-Right) = 2
  static constexpr uint8_t ROTATE_DIST[4][4] = {
      {0, 2, 1, 1},  // Up
      {2, 0, 1, 1},  // Down
      {1, 1, 0, 2},  // Left
      {1, 1, 2, 0}   // Right
  };
  if (arg1 < 4 && arg2 < 4) {
    return ROTATE_DIST[arg1][arg2];
  }
  return (arg1 == arg2) ? 0 : 1;
}

inline uint8_t calculate_attack_distance(uint8_t arg1, uint8_t arg2) {
  // 3x3 grid: convert to (row, col)
  if (arg1 > 8 || arg2 > 8) {
    return (arg1 == arg2) ? 0 : 1;
  }
  uint8_t r1 = arg1 / 3, c1 = arg1 % 3;
  uint8_t r2 = arg2 / 3, c2 = arg2 % 3;
  // Manhattan distance in the attack grid
  return std::abs(r1 - r2) + std::abs(c1 - c2);
}

inline uint8_t calculate_color_distance(uint8_t arg1, uint8_t arg2) {
  // ++, --, +=step, -=step
  if (arg1 > 3 || arg2 > 3) {
    return (arg1 == arg2) ? 0 : 1;
  }
  // Group by increment/decrement
  bool inc1 = (arg1 == 0 || arg1 == 2);  // ++ or +=step
  bool inc2 = (arg2 == 0 || arg2 == 2);
  return (inc1 == inc2) ? 1 : 2;  // Same direction = closer
}

class DistanceLUTBuilder {
public:
  struct ActionInfo {
    std::string name;
    uint8_t max_arg;
    uint8_t type_id;
    uint8_t start_offset;  // Where this action's encodings start
  };

  std::vector<ActionInfo> actions;
  std::array<std::array<uint8_t, 256>, 256> table;
  uint8_t total_encoded_actions = 0;

  DistanceLUTBuilder() {
    // Initialize all distances to maximum
    for (auto& row : table) {
      row.fill(15);  // Max distance
    }
  }

  // Register an action handler
  void register_action(uint8_t type_id, const std::string& name, uint8_t max_arg) {
    ActionInfo info;
    info.name = name;
    info.max_arg = max_arg;
    info.type_id = type_id;
    info.start_offset = total_encoded_actions;

    actions.push_back(info);
    total_encoded_actions += (max_arg + 1);
  }

  // Build the distance table after all actions are registered
  void build() {
    // Fill in distances for all valid action encodings
    for (uint8_t enc1 = 0; enc1 < total_encoded_actions && enc1 < 256; enc1++) {
      for (uint8_t enc2 = 0; enc2 < total_encoded_actions && enc2 < 256; enc2++) {
        table[enc1][enc2] = compute_distance(enc1, enc2);
      }
    }
  }

  // Convert (action_type, action_arg) to encoded value
  uint8_t encode_action(ActionType type, ActionArg arg) const {
    if (type >= actions.size()) return 0;  // Invalid -> NOOP
    const auto& action = actions[type];
    if (arg > action.max_arg) arg = action.max_arg;  // Clamp to valid range
    return action.start_offset + arg;
  }

  // Get distance from the table
  uint8_t get_distance(uint8_t enc1, uint8_t enc2) const {
    return table[enc1][enc2];
  }

private:
  // Decode encoded action back to (type, arg)
  std::pair<uint8_t, uint8_t> decode_action(uint8_t encoded) const {
    for (size_t type = 0; type < actions.size(); type++) {
      const auto& action = actions[type];
      uint8_t next_offset = (type + 1 < actions.size()) ? actions[type + 1].start_offset : total_encoded_actions;
      if (encoded < next_offset) {
        return {static_cast<uint8_t>(type), encoded - action.start_offset};
      }
    }
    return {0, 0};  // Invalid -> NOOP
  }

  uint8_t compute_distance(uint8_t enc1, uint8_t enc2) const {
    if (enc1 == enc2) return 0;

    auto [type1, arg1] = decode_action(enc1);
    auto [type2, arg2] = decode_action(enc2);

    if (type1 >= actions.size() || type2 >= actions.size()) {
      return 15;  // Invalid action
    }

    const auto& action1 = actions[type1];
    const auto& action2 = actions[type2];

    // Same action type - check for special intra-action distances
    if (type1 == type2) {
      const std::string& name = action1.name;

      if (name == "move") {
        return calculate_move_distance(arg1, arg2);
      } else if (name == "rotate") {
        return calculate_rotate_distance(arg1, arg2);
      } else if (name == "attack") {
        return calculate_attack_distance(arg1, arg2);
      } else if (name == "change_color") {
        return calculate_color_distance(arg1, arg2);
      } else if (name == "change_glyph") {
        return 1;  // All glyph changes are equally similar
      } else {
        return 0;  // Same action, same arg
      }
    }

    // Different action types - use group distances
    ActionGroup group1 = get_action_group(action1.name);
    ActionGroup group2 = get_action_group(action2.name);

    return GROUP_DISTANCES[group1][group2];
  }
};

// For integration with MettaGrid
class ActionDistanceLUT {
private:
  DistanceLUTBuilder builder;
  bool built = false;

public:
  // Register actions from MettaGrid's action handlers
  template <typename ActionHandlerContainer>
  void register_actions(const ActionHandlerContainer& handlers) {
    for (size_t i = 0; i < handlers.size(); i++) {
      const auto& handler = handlers[i];
      builder.register_action(static_cast<uint8_t>(i), handler->action_name(), handler->max_arg());
    }
    builder.build();
    built = true;
  }

  // Get distance between two actions
  uint8_t get_distance(ActionType type1, ActionArg arg1, ActionType type2, ActionArg arg2) const {
    if (!built) return 0;
    uint8_t enc1 = builder.encode_action(type1, arg1);
    uint8_t enc2 = builder.encode_action(type2, arg2);
    return builder.get_distance(enc1, enc2);
  }

  // Get encoded action value
  uint8_t encode_action(ActionType type, ActionArg arg) const {
    return builder.encode_action(type, arg);
  }

  // Get the full distance table for GPU upload
  void get_distance_table(uint8_t out[256][256]) const {
    for (int i = 0; i < 256; i++) {
      for (int j = 0; j < 256; j++) {
        out[i][j] = builder.table[i][j];
      }
    }
  }

  // Get total number of encoded actions
  uint8_t get_total_encoded_actions() const {
    return builder.total_encoded_actions;
  }
};

}  // namespace ActionDistance

#endif  // ACTION_DISTANCE_HPP_
