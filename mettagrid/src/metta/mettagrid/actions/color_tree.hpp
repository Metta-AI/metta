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

  ColorTreeActionConfig(const std::map<InventoryItem, InventoryQuantity>& required_resources,
                        const std::map<InventoryItem, InventoryQuantity>& consumed_resources,
                        const std::vector<uint8_t>& target_sequence,
                        float sequence_reward,
                        const std::map<uint8_t, InventoryItem>& color_to_item)
      : ActionConfig(required_resources, consumed_resources),
        target_sequence(target_sequence),
        sequence_reward(sequence_reward),
        color_to_item(color_to_item) {}
};

class ColorTree : public ActionHandler {
public:
  std::vector<uint8_t> _target_sequence;
  explicit ColorTree(const ColorTreeActionConfig& cfg)
      : ActionHandler(cfg, "color_tree"),
        _target_sequence(cfg.target_sequence),
        _sequence_reward(cfg.sequence_reward),
        _color_to_item(cfg.color_to_item),
        _current_sequence() {}

  unsigned char max_arg() const override {
    return 2;  // Only 3 color options: 0, 1, 2
  }

protected:
  bool _handle_action(Agent* actor, ActionArg arg) override {
    uint8_t color = static_cast<uint8_t>(arg);

    // Validate the color argument
    if (color > max_arg()) {
      return false;
    }

            // Add the color to the current sequence
    _current_sequence.push_back(color);
    actor->stats.add("color_tree.colors_added", 1.0f);

    // Find the corresponding inventory item for this color
    auto item_it = _color_to_item.find(color);
    if (item_it != _color_to_item.end()) {
      InventoryItem item = item_it->second;

      // Add the item to inventory for visualization
      InventoryDelta delta = actor->update_inventory(item, 1);
      if (delta <= 0) {
        // If we couldn't add the item (inventory full), we might want to handle this
        // For now, continue with the sequence checking
      }
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

      // Also clear inventory items after checking the window
      for (const auto& [mapped_color, mapped_item] : _color_to_item) {
        auto inv_it = actor->inventory.find(mapped_item);
        if (inv_it != actor->inventory.end()) {
          actor->update_inventory(mapped_item, -static_cast<InventoryDelta>(inv_it->second));
        }
      }
    }

    return true;
  }

private:

  float _sequence_reward;
  std::map<uint8_t, InventoryItem> _color_to_item;
  std::vector<uint8_t> _current_sequence;  // Tracks the current sequence of actions
};

#endif  // ACTIONS_COLOR_TREE_HPP_
