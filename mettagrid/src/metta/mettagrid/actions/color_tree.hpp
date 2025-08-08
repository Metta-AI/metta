#ifndef ACTIONS_COLOR_TREE_HPP_
#define ACTIONS_COLOR_TREE_HPP_

#include <string>
#include <vector>
#include <map>
#include <array>
#include <cstring>
#include <limits>

#include "action_handler.hpp"
#include "objects/agent.hpp"
#include "types.hpp"

enum class ColorTreeRewardMode : uint8_t {
  PRECISE = 0,  // All or nothing reward
  PARTIAL = 1,  // Proportional to correct positions
  DENSE = 2     // Immediate reward for correct actions
};

// Helper function to convert string to reward mode enum
inline ColorTreeRewardMode string_to_reward_mode(const std::string& mode_str) {
  if (mode_str == "precise") return ColorTreeRewardMode::PRECISE;
  if (mode_str == "partial") return ColorTreeRewardMode::PARTIAL;
  if (mode_str == "dense") return ColorTreeRewardMode::DENSE;
  throw std::runtime_error("Invalid reward mode. Must be 'precise', 'partial', or 'dense'");
}

struct ColorTreeActionConfig : public ActionConfig {
  std::vector<uint8_t> target_sequence;  // Target color sequence to match
  float sequence_reward;                 // Reward given for correct sequence match
  std::map<uint8_t, InventoryItem> color_to_item;  // Maps color values to inventory items
  size_t num_trials;                     // Number of different sequences to test
  std::vector<std::vector<uint8_t>> trial_sequences;  // Different sequences for each trial
  size_t attempts_per_trial;             // Number of attempts allowed per trial
  ColorTreeRewardMode reward_mode;       // Reward mode enum

  ColorTreeActionConfig(const std::map<InventoryItem, InventoryQuantity>& required_resources,
                        const std::map<InventoryItem, InventoryQuantity>& consumed_resources,
                        const std::vector<uint8_t>& target_sequence,
                        float sequence_reward,
                        const std::map<uint8_t, InventoryItem>& color_to_item,
                        int num_trials = 1,
                        const std::vector<std::vector<uint8_t>>& trial_sequences = {},
                        int attempts_per_trial = 4,
                        const std::string& reward_mode_str = "precise")
      : ActionConfig(required_resources, consumed_resources),
        target_sequence(target_sequence),
        sequence_reward(sequence_reward),
        color_to_item(color_to_item),
        num_trials(static_cast<size_t>(num_trials)),
        trial_sequences(trial_sequences),
        attempts_per_trial(static_cast<size_t>(attempts_per_trial)),
        reward_mode(string_to_reward_mode(reward_mode_str)) {
    // Validate trial sequences have same length as target sequence
    for (const auto& seq : trial_sequences) {
      if (!seq.empty() && seq.size() != target_sequence.size()) {
        throw std::runtime_error("All trial sequences must have the same length as target sequence");
      }
    }
  }
};

class ColorTree : public ActionHandler {
public:
  explicit ColorTree(const ColorTreeActionConfig& cfg)
      : ActionHandler(cfg, "color_tree"),
        _base_target_sequence(cfg.target_sequence),
        _sequence_reward(cfg.sequence_reward),
        _color_to_item(cfg.color_to_item),
        _num_trials(cfg.num_trials),
        _trial_sequences(cfg.trial_sequences),
        _attempts_per_trial(cfg.attempts_per_trial),
        _reward_mode(cfg.reward_mode),
        _max_sequence_size(cfg.target_sequence.size()),
        _actions_per_trial(_attempts_per_trial * _max_sequence_size),
        _per_position_reward(_sequence_reward / static_cast<float>(_max_sequence_size)) {
    // Precompute fast lookup table for color_to_item and max color value
    if (!_color_to_item.empty()) {
      _max_color = _color_to_item.rbegin()->first;
      _color_to_item_fast.assign(static_cast<size_t>(_max_color) + 1, INVALID_ITEM);
      for (const auto& [color, item] : _color_to_item) {
        if (static_cast<size_t>(color) >= _color_to_item_fast.size()) continue;
        _color_to_item_fast[static_cast<size_t>(color)] = item;
      }
    } else {
      _max_color = 0;
      _color_to_item_fast.assign(1, INVALID_ITEM);
    }

    // Initialize global target pointer
    if (_num_trials > 1 && !_trial_sequences.empty()) {
      _current_target_ptr_global = &_trial_sequences[0];
    } else {
      _current_target_ptr_global = &_base_target_sequence;
    }
  }

  unsigned char max_arg() const override {
    // Return the maximum color value that can be used
    return static_cast<unsigned char>(_max_color);
  }

protected:
  bool _handle_action(Agent* actor, ActionArg arg) override {
    uint8_t color = static_cast<uint8_t>(arg);

    // Validate the color argument
    if (color > max_arg()) {
      return false;
    }

    // Get or create per-agent state (indexed by agent_id for O(1) access)
    const size_t agent_id = static_cast<size_t>(actor->agent_id);
    if (agent_id >= _agent_state.size()) {
      size_t old_size = _agent_state.size();
      _agent_state.resize(agent_id + 1);
      // Initialize new entries
      for (size_t i = old_size; i <= agent_id; ++i) {
        _agent_state[i] = PerAgentData();
      }
    }
    auto& agent_data = _agent_state[agent_id];

    // Global trial switching based on total action calls normalized by number of agents (sync all agents)
    if (_agents_per_step == 0) {
      // We cannot query env here cheaply; initialize to actor->agent_id+1 on first encounter and let it grow conservatively
      _agents_per_step = static_cast<size_t>(actor->agent_id) + 1;
    } else if (static_cast<size_t>(actor->agent_id) + 1 > _agents_per_step) {
      _agents_per_step = static_cast<size_t>(actor->agent_id) + 1;
    }
    _global_action_count += 1;
    if (_global_action_count % (_actions_per_trial * _agents_per_step) == 0) {
      _current_trial_global += 1;
      if (_num_trials > 1 && _current_trial_global < _num_trials && _current_trial_global < _trial_sequences.size()) {
        _current_target_ptr_global = &_trial_sequences[_current_trial_global];
      } else {
        _current_target_ptr_global = &_base_target_sequence;
      }
      actor->stats.add("color_tree.trial_started", 1.0f);
    }

    // Update correctness bit for this position, then advance index
    size_t write_index = agent_data.position_in_sequence;
    bool is_correct = (color == (*_current_target_ptr_global)[write_index]);
    uint8_t bit_mask = static_cast<uint8_t>(1u << write_index);
    bool prev_bit = (agent_data.correctness_mask & bit_mask) != 0;
    if (prev_bit != is_correct) {
      agent_data.correct_positions_count += is_correct ? 1 : -1;
      if (is_correct) {
        agent_data.correctness_mask |= bit_mask;
      } else {
        agent_data.correctness_mask &= static_cast<uint8_t>(~bit_mask);
      }
    }
    // Advance write index
    write_index++;
    if (write_index == _max_sequence_size) {
      write_index = 0;
      agent_data.window_filled = true;
    }
    agent_data.position_in_sequence = write_index;
    actor->stats.add("color_tree.colors_added", 1.0f);

    // Resolve the target sequence for current global trial
    const std::vector<uint8_t>& current_target = *_current_target_ptr_global;

    // Dense reward mode: give immediate reward for correct position
    if (_reward_mode == ColorTreeRewardMode::DENSE) {
      size_t compare_index = (agent_data.position_in_sequence == 0)
                                 ? (_max_sequence_size - 1)
                                 : (agent_data.position_in_sequence - 1);
      if (color == (*_current_target_ptr_global)[compare_index]) {
        *actor->reward += _per_position_reward;
        actor->stats.add("color_tree.correct_position", 1.0f);
      }
    }

    // Find the corresponding inventory item for this color (O(1) lookup)
    InventoryItem new_item = INVALID_ITEM;
    if (static_cast<size_t>(color) < _color_to_item_fast.size()) {
      new_item = _color_to_item_fast[static_cast<size_t>(color)];
    }
    if (new_item == INVALID_ITEM) {
      // Invalid color - this shouldn't happen due to earlier validation
      return false;
    }

    // Inventory visualization: only touch inventory if the item actually changes
    if (new_item != agent_data.current_item) {
      // Remove one of the previously stored item if we had set one
      if (agent_data.current_item != INVALID_ITEM) {
        actor->update_inventory(agent_data.current_item, -static_cast<InventoryDelta>(1));
      }
      // Add the new item
      agent_data.current_item = new_item;
      if (actor->update_inventory(agent_data.current_item, 1) <= 0) {
        actor->stats.add("color_tree.inventory_full", 1.0f);
      }
    }

        // Check if we've completed a fixed window
    if (agent_data.window_filled) {
      agent_data.window_filled = false;
      if (_reward_mode == ColorTreeRewardMode::PRECISE) {
        if (agent_data.correct_positions_count == static_cast<int>(_max_sequence_size)) {
          *actor->reward += _sequence_reward;
          actor->stats.add("color_tree.sequence_completed", 1.0f);
        }
      } else if (_reward_mode == ColorTreeRewardMode::PARTIAL) {
        int correct_positions = agent_data.correct_positions_count;
        if (correct_positions > 0) {
          float partial_reward = _sequence_reward *
                                 (static_cast<float>(correct_positions) / static_cast<float>(_max_sequence_size));
          *actor->reward += partial_reward;
          actor->stats.add("color_tree.partial_matches", static_cast<float>(correct_positions));
          if (correct_positions == static_cast<int>(_max_sequence_size)) {
            actor->stats.add("color_tree.sequence_completed", 1.0f);
          }
        }
      } else if (_reward_mode == ColorTreeRewardMode::DENSE) {
        if (agent_data.correct_positions_count == static_cast<int>(_max_sequence_size)) {
          actor->stats.add("color_tree.sequence_completed", 1.0f);
        }
      }
      // Reset window correctness for next window
      agent_data.correctness_mask = 0;
      agent_data.correct_positions_count = 0;
    }

    return true;
  }

private:
  struct PerAgentData {
    InventoryItem current_item = INVALID_ITEM; // Item currently in inventory
    size_t position_in_sequence = 0; // Write index within the fixed window
    bool window_filled = false;      // Whether we've written a full window
    uint8_t correctness_mask = 0;    // Up to 8 positions (we use lower bits)
    int correct_positions_count = 0; // Running count in current window
  };

  // Per-agent state indexed by agent_id for O(1) access
  std::vector<PerAgentData> _agent_state;

  static constexpr InventoryItem INVALID_ITEM = static_cast<InventoryItem>(-1);

  std::vector<uint8_t> _base_target_sequence;  // Default target sequence
  float _sequence_reward;
  std::map<uint8_t, InventoryItem> _color_to_item;
  std::vector<InventoryItem> _color_to_item_fast;   // Direct index lookup for colors
  size_t _max_color = 0;                            // Cached maximum color id
  size_t _max_sequence_size{};                  // Shared length of sequences
  size_t _num_trials;
  std::vector<std::vector<uint8_t>> _trial_sequences;
  size_t _attempts_per_trial;
  ColorTreeRewardMode _reward_mode;
  size_t _actions_per_trial{};                  // Pre-calculated: _attempts_per_trial * _max_sequence_size
  float _per_position_reward;
  // Global trial state (sync across agents)
  size_t _global_action_count = 0;
  size_t _agents_per_step = 0;
  size_t _current_trial_global = 0;
  const std::vector<uint8_t>* _current_target_ptr_global = nullptr;
};

#endif  // ACTIONS_COLOR_TREE_HPP_
