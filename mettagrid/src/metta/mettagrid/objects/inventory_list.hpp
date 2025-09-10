#ifndef OBJECTS_INVENTORY_LIST_HPP_
#define OBJECTS_INVENTORY_LIST_HPP_

#include <algorithm>
#include <map>
#include <random>
#include <vector>

#include "constants.hpp"
#include "event.hpp"

// Forward declaration
class EventManager;

// Resource instance tracking for stochastic loss
struct ResourceInstance {
  uint64_t id;  // Unique identifier for this resource instance
  InventoryItem item_type;
  unsigned int creation_timestep;
};

class InventoryList {
public:
  // Constructor without resource loss
  InventoryList() : next_resource_id(1) {}

  // Constructor with resource loss
  InventoryList(EventManager* event_manager, std::mt19937* rng,
                const std::map<InventoryItem, float>& resource_loss_prob)
      : next_resource_id(1), event_manager(event_manager), rng(rng),
        resource_loss_prob(resource_loss_prob) {}

  // Core inventory management
  std::map<InventoryItem, InventoryQuantity> inventory;

  // Resource instance tracking
  std::map<uint64_t, ResourceInstance> resource_instances;
  std::map<InventoryItem, std::vector<uint64_t>> item_to_resources;
  uint64_t next_resource_id;

  // Resource loss configuration
  std::map<InventoryItem, float> resource_loss_prob;

  // RNG and event manager (set by parent class)
  std::mt19937* rng = nullptr;
  EventManager* event_manager = nullptr;

  // Getter for resource loss probabilities
  const std::map<InventoryItem, float>& get_resource_loss_prob() const {
    return resource_loss_prob;
  }

  // Populate initial inventory and initialize resource instances
  void populate_initial_inventory(const std::map<InventoryItem, InventoryQuantity>& initial_inventory, GridObjectId object_id) {
    for (const auto& [item, amount] : initial_inventory) {
      if (amount > 0) {
        this->inventory[item] = amount;
      }
    }

    // Initialize resource instances if we have both event manager and RNG
    if (this->event_manager && this->rng) {
      this->initialize_resource_instances(object_id);
    }
  }

  // Resource instance management
  inline uint64_t create_resource_instance(InventoryItem item_type, unsigned int current_timestep) {
    uint64_t resource_id = next_resource_id++;
    resource_instances[resource_id] = {resource_id, item_type, current_timestep};
    item_to_resources[item_type].push_back(resource_id);
    return resource_id;
  }

  inline void remove_resource_instance(uint64_t resource_id) {
    auto it = resource_instances.find(resource_id);
    if (it != resource_instances.end()) {
      InventoryItem item_type = it->second.item_type;
      resource_instances.erase(it);

      // Remove from item_to_resources
      auto& resources = item_to_resources[item_type];
      resources.erase(std::remove(resources.begin(), resources.end(), resource_id), resources.end());
      if (resources.empty()) {
        item_to_resources.erase(item_type);
      }
    }
  }

  inline uint64_t get_random_resource_id(InventoryItem item_type) {
    auto it = item_to_resources.find(item_type);
    if (it == item_to_resources.end() || it->second.empty()) {
      return 0;  // No resources of this type
    }

    // Use proper RNG if available, otherwise fall back to rand()
    if (this->rng) {
      std::uniform_int_distribution<size_t> dist(0, it->second.size() - 1);
      size_t index = dist(*this->rng);
      return it->second[index];
    } else {
      size_t index = rand() % it->second.size();
      return it->second[index];
    }
  }

  // Helper method to create resource instances and schedule loss events
  void create_and_schedule_resources(InventoryItem item_type, int count, unsigned int current_timestep,
                                   GridObjectId object_id) {
    if (!this->event_manager || !this->rng) {
      return;
    }
    auto prob_it = this->resource_loss_prob.find(item_type);
    if (prob_it == this->resource_loss_prob.end() || prob_it->second <= 0.0f) {
      for (int i = 0; i < count; i++) {
        create_resource_instance(item_type, current_timestep);
      }
      return;
    }
    for (int i = 0; i < count; i++) {
      uint64_t resource_id = create_resource_instance(item_type, current_timestep);
      std::exponential_distribution<float> exp_dist(prob_it->second);
      float lifetime = exp_dist(*this->rng);
      unsigned int loss_timestep = current_timestep + static_cast<unsigned int>(std::ceil(lifetime));
      this->event_manager->schedule_event(EventType::StochasticResourceLoss,
                                        loss_timestep - current_timestep,
                                        object_id, // Use object's GridObjectId
                                        static_cast<EventArg>(resource_id));
    }
  }

  // Inventory update with stochastic resource loss support
  InventoryDelta update_inventory(InventoryItem item, InventoryDelta attempted_delta,
                                GridObjectId object_id) {
    InventoryQuantity initial_amount = this->inventory[item];
    int new_amount = static_cast<int>(initial_amount + attempted_delta);

    constexpr int min = std::numeric_limits<InventoryQuantity>::min();
    constexpr int max = std::numeric_limits<InventoryQuantity>::max();
    InventoryQuantity clamped_amount = static_cast<InventoryQuantity>(std::clamp(new_amount, min, max));

    if (clamped_amount == 0) {
      this->inventory.erase(item);
    } else {
      this->inventory[item] = clamped_amount;
    }

    InventoryDelta delta = clamped_amount - initial_amount;

    if (delta != 0) {
      if (delta > 0) {
        // Create resource instances and schedule loss events for added items
        if (this->event_manager && this->rng) {
          unsigned int current_timestep = this->event_manager->get_current_timestep();
          create_and_schedule_resources(item, delta, current_timestep, object_id);
        }
      } else {
        // Remove random resource instances for removed items
        for (int i = 0; i < -delta; i++) {
          uint64_t resource_id = get_random_resource_id(item);
          if (resource_id != 0) {
            remove_resource_instance(resource_id);
          }
        }
      }
    }

    return delta;
  }

  // Initialize resource instances for existing inventory
  void initialize_resource_instances(GridObjectId object_id) {
    if (this->event_manager && this->rng) {
      unsigned int current_timestep = this->event_manager->get_current_timestep();
      for (const auto& [item, amount] : this->inventory) {
        if (amount > 0) {
          create_and_schedule_resources(item, amount, current_timestep, object_id);
        }
      }
    }
  }

  // Clear all resources (useful for cleanup)
  void clear() {
    inventory.clear();
    resource_instances.clear();
    item_to_resources.clear();
    next_resource_id = 1;
  }
};

#endif  // OBJECTS_INVENTORY_LIST_HPP_
