#ifndef OBJECTS_PRODUCTION_HANDLER_HPP_
#define OBJECTS_PRODUCTION_HANDLER_HPP_

#include "../event.hpp"
#include "../grid.hpp"
#include "constants.hpp"
#include "converter.hpp"
#include "agent.hpp"

// Handles the FinishConverting event
class ProductionHandler : public EventHandler {
public:
  explicit ProductionHandler(EventManager* event_manager) : EventHandler(event_manager) {}

  void handle_event(GridObjectId obj_id, EventArg /*arg*/) override {
    Converter* converter = static_cast<Converter*>(this->event_manager->grid->object(obj_id));
    if (!converter) {
      return;
    }

    converter->finish_converting();
    converter->stats.incr(converter->type_name + ".produced");
  }
};

// Handles the CoolDown event
class CoolDownHandler : public EventHandler {
public:
  explicit CoolDownHandler(EventManager* event_manager) : EventHandler(event_manager) {}

  void handle_event(GridObjectId obj_id, EventArg /*arg*/) override {
    Converter* converter = static_cast<Converter*>(this->event_manager->grid->object(obj_id));
    if (!converter) {
      return;
    }

    converter->finish_cooldown();
  }
};

// Handles the StochasticResourceLoss event
class StochasticResourceLossHandler : public EventHandler {
public:
  explicit StochasticResourceLossHandler(EventManager* event_manager) : EventHandler(event_manager) {}

  void handle_event(GridObjectId obj_id, EventArg arg) override {
    // The arg contains the resource_id
    uint64_t resource_id = static_cast<uint64_t>(arg);

    // Get the agent directly from the grid using the obj_id
    Agent* agent = dynamic_cast<Agent*>(this->event_manager->grid->object(obj_id));
    if (!agent) {
      return;
    }

    // Check if this agent has the resource
    auto it = agent->resource_instances.find(resource_id);
    if (it != agent->resource_instances.end()) {
      // Found the resource - remove it
      InventoryItem item_type = it->second.item_type;

      // Remove the resource instance
      agent->inventory_list.remove_resource_instance(resource_id);

      // Update the inventory count
      if (agent->inventory_list.inventory.count(item_type) > 0) {
        agent->inventory_list.inventory[item_type]--;
        if (agent->inventory_list.inventory[item_type] == 0) {
          agent->inventory_list.inventory.erase(item_type);
        }
      }

      // Update stats
      agent->stats.add(agent->stats.resource_name(item_type) + ".lost", 1);
    }

    // Resource not found - it may have already been removed by another mechanism
    // This is expected behavior, so we just return silently
  }
};

#endif  // OBJECTS_PRODUCTION_HANDLER_HPP_
