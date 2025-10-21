#include <iostream>
#include <memory>
#include <vector>

#include "cpp/include/mettagrid/objects/agent.hpp"
#include "cpp/include/mettagrid/objects/agent_config.hpp"
#include "cpp/include/mettagrid/objects/has_inventory.hpp"
#include "cpp/include/mettagrid/objects/inventory_config.hpp"

int main() {
  // Test that negative delta is evenly distributed among agents
  InventoryConfig inv_cfg;
  inv_cfg.limits = {{{0}, 100}};  // ORE = 0, limit of 100

  // Create agent configuration
  AgentConfig agent_cfg(1, "test_agent");
  agent_cfg.inventory_config = inv_cfg;

  // Create agents with the inventory configuration
  Agent agent1(0, 0, agent_cfg);
  Agent agent2(1, 0, agent_cfg);
  Agent agent3(2, 0, agent_cfg);

  // Pre-fill agent inventories with 20 ore each
  agent1.update_inventory(0, 20);
  agent2.update_inventory(0, 20);
  agent3.update_inventory(0, 20);

  std::cout << "Before shared_update:" << std::endl;
  std::cout << "agent1 ore: " << (int)agent1.inventory.amount(0) << std::endl;
  std::cout << "agent2 ore: " << (int)agent2.inventory.amount(0) << std::endl;
  std::cout << "agent3 ore: " << (int)agent3.inventory.amount(0) << std::endl;

  // Create vector of HasInventory pointers (agents inherit from HasInventory)
  std::vector<HasInventory*> inventory_havers = {&agent1, &agent2, &agent3};

  // Remove 30 ore total, should remove 10 from each agent
  InventoryDelta consumed = HasInventory::shared_update(inventory_havers, 0, -30);

  std::cout << "\nAfter shared_update(-30):" << std::endl;
  std::cout << "consumed: " << consumed << std::endl;
  std::cout << "agent1 ore: " << (int)agent1.inventory.amount(0) << std::endl;
  std::cout << "agent2 ore: " << (int)agent2.inventory.amount(0) << std::endl;
  std::cout << "agent3 ore: " << (int)agent3.inventory.amount(0) << std::endl;

  // Test positive delta (adding resources)
  std::cout << "\n--- Testing positive delta ---" << std::endl;

  // Add 15 ore total, should add 5 to each agent
  InventoryDelta added = HasInventory::shared_update(inventory_havers, 0, 15);

  std::cout << "After shared_update(+15):" << std::endl;
  std::cout << "added: " << added << std::endl;
  std::cout << "agent1 ore: " << (int)agent1.inventory.amount(0) << std::endl;
  std::cout << "agent2 ore: " << (int)agent2.inventory.amount(0) << std::endl;
  std::cout << "agent3 ore: " << (int)agent3.inventory.amount(0) << std::endl;

  // Test with uneven distribution
  std::cout << "\n--- Testing uneven distribution ---" << std::endl;

  // Set different amounts
  agent1.update_inventory(0, -agent1.inventory.amount(0) + 5);   // Set to 5
  agent2.update_inventory(0, -agent2.inventory.amount(0) + 10);  // Set to 10
  agent3.update_inventory(0, -agent3.inventory.amount(0) + 30);  // Set to 30

  std::cout << "Before uneven test:" << std::endl;
  std::cout << "agent1 ore: " << (int)agent1.inventory.amount(0) << std::endl;
  std::cout << "agent2 ore: " << (int)agent2.inventory.amount(0) << std::endl;
  std::cout << "agent3 ore: " << (int)agent3.inventory.amount(0) << std::endl;

  // Try to remove 20 ore - agent1 can only give 5, agent2 can give 10, agent3 gives 5
  consumed = HasInventory::shared_update(inventory_havers, 0, -20);

  std::cout << "\nAfter shared_update(-20) with uneven distribution:" << std::endl;
  std::cout << "consumed: " << consumed << std::endl;
  std::cout << "agent1 ore: " << (int)agent1.inventory.amount(0) << std::endl;
  std::cout << "agent2 ore: " << (int)agent2.inventory.amount(0) << std::endl;
  std::cout << "agent3 ore: " << (int)agent3.inventory.amount(0) << std::endl;

  return 0;
}
