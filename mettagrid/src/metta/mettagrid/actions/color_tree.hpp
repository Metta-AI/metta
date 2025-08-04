#ifndef ACTIONS_COLOR_TREE_HPP_
#define ACTIONS_COLOR_TREE_HPP_

#include <string>
#include <vector>
#include <map>
#include <array>

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
        _current_sequence_size(0),
        _action_count(0),
        _current_trial(0),
        _current_inventory_item(INVALID_ITEM) {
    // Initialize target sequence for first trial
    _update_target_sequence();

    // Pre-calculate actions per trial
    _actions_per_trial = _attempts_per_trial * _target_sequence.size();

    // Reserve space for maximum sequence size
    if (!_target_sequence.empty()) {
      _max_sequence_size = _target_sequence.size();
      _current_sequence.resize(_max_sequence_size);
    }
  }

  unsigned char max_arg() const override {
    // Return the maximum color value that can be used
    if (_color_to_item.empty()) {
      return 0;
    }
    return _color_to_item.rbegin()->first;  // Highest key in the map
  }

protected:
  bool _handle_action(Agent* actor, ActionArg arg) override {
    uint8_t color = static_cast<uint8_t>(arg);

    // Validate the color argument
    if (color > max_arg()) {
      return false;
    }

    // Track total actions for trial switching
    _action_count++;

    // Check if we need to switch to a new trial
    if (_action_count % _actions_per_trial == 0 && _action_count > 0) {
      _current_trial++;
      if (_current_trial < _num_trials) {
        _update_target_sequence();
        actor->stats.add("color_tree.trial_started", 1.0f);
      }
    }

    // Add the color to the current sequence
    size_t position_in_sequence = _current_sequence_size % _max_sequence_size;
    _current_sequence[position_in_sequence] = color;
    _current_sequence_size++;
    actor->stats.add("color_tree.colors_added", 1.0f);

    // Dense reward mode: give immediate reward for correct position
    if (_reward_mode == ColorTreeRewardMode::DENSE) {
      if (color == _target_sequence[position_in_sequence]) {
        float position_reward = _sequence_reward / _target_sequence.size();
        *actor->reward += position_reward;
        actor->stats.add("color_tree.correct_position", 1.0f);
      }
    }

    // Clear only the previously stored item (if any)
    if (_current_inventory_item != INVALID_ITEM) {
      auto inv_it = actor->inventory.find(_current_inventory_item);
      if (inv_it != actor->inventory.end() && inv_it->second > 0) {
        actor->update_inventory(_current_inventory_item, -static_cast<InventoryDelta>(inv_it->second));
      }
    }

    // Find the corresponding inventory item for this color
    auto item_it = _color_to_item.find(color);
    if (item_it == _color_to_item.end()) {
      // Invalid color - this shouldn't happen due to earlier validation
      return false;
    }

    // Add the item to inventory for visualization
    _current_inventory_item = item_it->second;
    InventoryDelta delta = actor->update_inventory(_current_inventory_item, 1);
    if (delta <= 0) {
      // Log warning but continue - visualization may be incomplete
      actor->stats.add("color_tree.inventory_full", 1.0f);
    }

        // Check if we've completed a fixed window
    if (_current_sequence_size % _max_sequence_size == 0 && _current_sequence_size > 0) {
      if (_reward_mode == ColorTreeRewardMode::PRECISE) {
        // Precise mode: all or nothing
        bool sequence_matches = true;
        for (size_t i = 0; i < _target_sequence.size(); ++i) {
          if (_current_sequence[i] != _target_sequence[i]) {
            sequence_matches = false;
            break;
          }
        }

        if (sequence_matches) {
          *actor->reward += _sequence_reward;
          actor->stats.add("color_tree.sequence_completed", 1.0f);
        }
      } else if (_reward_mode == ColorTreeRewardMode::PARTIAL) {
        // Partial mode: reward proportional to correct positions
        size_t correct_positions = 0;
        for (size_t i = 0; i < _target_sequence.size(); ++i) {
          if (_current_sequence[i] == _target_sequence[i]) {
            correct_positions++;
          }
        }

        if (correct_positions > 0) {
          float partial_reward = _sequence_reward * (static_cast<float>(correct_positions) / _target_sequence.size());
          *actor->reward += partial_reward;
          actor->stats.add("color_tree.partial_matches", static_cast<float>(correct_positions));

          if (correct_positions == _target_sequence.size()) {
            actor->stats.add("color_tree.sequence_completed", 1.0f);
          }
        }
      }
      // Note: dense mode already gave rewards during each action
    }

    return true;
  }

private:
  void _update_target_sequence() {
    if (_num_trials > 1 && _current_trial < _trial_sequences.size() && _current_trial < _num_trials) {
      _target_sequence = _trial_sequences[_current_trial];
    } else {
      _target_sequence = _base_target_sequence;
    }
    // Update max sequence size and actions per trial if sequence changes
    _max_sequence_size = _target_sequence.size();
    _actions_per_trial = _attempts_per_trial * _max_sequence_size;
  }

  static constexpr InventoryItem INVALID_ITEM = static_cast<InventoryItem>(-1);

  std::vector<uint8_t> _base_target_sequence;  // Default target sequence
  std::vector<uint8_t> _target_sequence;        // Current active target sequence
  float _sequence_reward;
  std::map<uint8_t, InventoryItem> _color_to_item;
  std::vector<uint8_t> _current_sequence;       // Fixed-size buffer for current sequence
  size_t _current_sequence_size;                // Actual number of actions in current sequence
  size_t _max_sequence_size;                    // Maximum sequence size
  size_t _num_trials;
  std::vector<std::vector<uint8_t>> _trial_sequences;
  size_t _action_count;
  size_t _current_trial;
  size_t _attempts_per_trial;
  size_t _actions_per_trial;                    // Pre-calculated: attempts * sequence_size
  ColorTreeRewardMode _reward_mode;             // Reward mode enum
  InventoryItem _current_inventory_item;        // Track which item is currently in inventory
};

#endif  // ACTIONS_COLOR_TREE_HPP_
