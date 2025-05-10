#ifndef ATTACK_NEAREST_HPP
#define ATTACK_NEAREST_HPP

#include <cstdint>
#include <string>

#include "attack.hpp"
#include "constants.hpp"
#include "objects/agent.hpp"

namespace Actions {

class AttackNearest : public Attack {
public:
  AttackNearest(const ActionConfig& cfg) : Attack(cfg, "attack_nearest") {}

  uint8_t max_arg() const override {
    return 0;
  }

  ActionHandler* clone() const override {
    return new AttackNearest(*this);
  }

protected:
  bool _handle_action(uint32_t actor_id, Agent* actor, c_actions_type arg) override {
    // Check if agent has lasers
    if (actor->inventory[InventoryItem::laser] == 0) {
      return false;
    }

    actor->update_inventory(InventoryItem::laser, -1);

    // Use the validate_orientation utility method
    validate_orientation(actor);

    // Scan the space to find the nearest agent. Prefer the middle (offset 0) before the edges (offset -1, 1).
    for (int32_t distance = 1; distance < 4; distance++) {
      for (int32_t i = 0; i < 3; i++) {
        int32_t offset = i;
        if (offset == 2) {
          // Sort of a mod 3 operation.
          offset = -1;
        }

        GridLocation target_loc = _grid->relative_location(
            actor->location, static_cast<Orientation>(actor->orientation), static_cast<int16_t>(distance), offset);

        // Use is_valid_location utility method
        if (!is_valid_location(target_loc)) {
          continue;  // Skip this location and try the next one
        }

        target_loc.layer = GridLayer::Agent_Layer;
        GridObject* obj = safe_object_at(target_loc);

        Agent* agent_target = nullptr;
        if (obj != nullptr) {
          agent_target = dynamic_cast<Agent*>(obj);
          if (obj != nullptr && agent_target == nullptr) {
            throw std::runtime_error("Object at target location is not an Agent");
          }
        }

        if (agent_target) {
          return _handle_target(actor_id, actor, target_loc);
        }
      }
    }

    return false;
  }
};
}  // namespace Actions
#endif  // ATTACK_NEAREST_HPP