#ifndef ACTIONS_COLOR_TREE_HPP_
#define ACTIONS_COLOR_TREE_HPP_

#include <string>
#include <vector>
#include <map>

#include "action_handler.hpp"
#include "objects/agent.hpp"
#include "types.hpp"

struct ColorTreeActionConfig : public ActionConfig {
  std::vector<uint8_t> target_sequence;  // Target color sequence to match
  float sequence_reward;                 // Reward given for correct sequence match
  std::map<uint8_t, InventoryItem> color_to_item;  // Maps color values to inventory items
  size_t num_trials;                     // Number of different sequences to test
  std::vector<std::vector<uint8_t>> trial_sequences;  // Different sequences for each trial
  size_t attempts_per_trial;             // Number of attempts allowed per trial

  ColorTreeActionConfig(const std::map<InventoryItem, InventoryQuantity>& required_resources,
                        const std::map<InventoryItem, InventoryQuantity>& consumed_resources,
                        const std::vector<uint8_t>& target_sequence,
                        float sequence_reward,
                        const std::map<uint8_t, InventoryItem>& color_to_item,
                        int num_trials = 1,
                        const std::vector<std::vector<uint8_t>>& trial_sequences = {},
                        int attempts_per_trial = 4)
      : ActionConfig(required_resources, consumed_resources),
        target_sequence(target_sequence),
        sequence_reward(sequence_reward),
        color_to_item(color_to_item),
        num_trials(static_cast<size_t>(num_trials)),
        trial_sequences(trial_sequences),
        attempts_per_trial(static_cast<size_t>(attempts_per_trial)) {}
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
        _current_sequence(),
        _action_count(0),
        _current_trial(0) {
    // Initialize target sequence for first trial
    _update_target_sequence();
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
    size_t actions_per_trial = _attempts_per_trial * _target_sequence.size();
    if (_action_count % actions_per_trial == 0 && _action_count > 0) {
      _current_trial++;
      if (_current_trial < _num_trials) {
        _update_target_sequence();
        actor->stats.add("color_tree.trial_started", 1.0f);
      }
    }

    // Add the color to the current sequence
    _current_sequence.push_back(color);
    actor->stats.add("color_tree.colors_added", 1.0f);

    // Clear all color items from inventory before adding the new one
    for (const auto& [mapped_color, mapped_item] : _color_to_item) {
      auto inv_it = actor->inventory.find(mapped_item);
      if (inv_it != actor->inventory.end()) {
        actor->update_inventory(mapped_item, -static_cast<InventoryDelta>(inv_it->second));
      }
    }

    // Find the corresponding inventory item for this color
    auto item_it = _color_to_item.find(color);
    if (item_it != _color_to_item.end()) {
      InventoryItem item = item_it->second;

      // Add the item to inventory for visualization
      InventoryDelta delta = actor->update_inventory(item, 1);
      if (delta <= 0) {
        // Log warning but continue - visualization may be incomplete
        actor->stats.add("color_tree.inventory_full", 1.0f);
      }
    } else {
      // Invalid color - this shouldn't happen due to earlier validation
      return false;
    }

    // Check if we've completed a fixed window
    if (_current_sequence.size() == _target_sequence.size()) {
      // Check if the current fixed window matches the target sequence
      bool sequence_matches = true;
      for (size_t i = 0; i < _target_sequence.size(); ++i) {
        if (_current_sequence[i] != _target_sequence[i]) {
          sequence_matches = false;
          break;
        }
      }

      if (sequence_matches) {
        // Give reward for completing the sequence
        *actor->reward += _sequence_reward;
        actor->stats.add("color_tree.sequence_completed", 1.0f);
      }

      // Clear the sequence tracker
      _current_sequence.clear();
      // Note: Inventory is already cleared before each action, so no need to clear here
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
  }

  std::vector<uint8_t> _base_target_sequence;  // Default target sequence
  std::vector<uint8_t> _target_sequence;        // Current active target sequence
  float _sequence_reward;
  std::map<uint8_t, InventoryItem> _color_to_item;
  std::vector<uint8_t> _current_sequence;       // Tracks the current sequence of actions
  size_t _num_trials;
  std::vector<std::vector<uint8_t>> _trial_sequences;
  size_t _action_count;
  size_t _current_trial;
  size_t _attempts_per_trial;
};

#endif  // ACTIONS_COLOR_TREE_HPP_
