#include <iostream>
#include <vector>
#include "cpp/include/mettagrid/objects/inventory.hpp"
#include "cpp/include/mettagrid/objects/inventory_config.hpp"

int main() {
    // Test that negative delta is evenly distributed among inventories
    InventoryConfig cfg;
    cfg.limits = {{{0}, 100}};  // ORE = 0
    
    Inventory inv1(cfg);
    Inventory inv2(cfg);
    Inventory inv3(cfg);
    
    // Pre-fill inventories with 20 ore each
    inv1.update(0, 20);
    inv2.update(0, 20);
    inv3.update(0, 20);
    
    std::cout << "Before shared_update:" << std::endl;
    std::cout << "inv1: " << (int)inv1.amount(0) << std::endl;
    std::cout << "inv2: " << (int)inv2.amount(0) << std::endl;
    std::cout << "inv3: " << (int)inv3.amount(0) << std::endl;
    
    std::vector<Inventory*> inventories = {&inv1, &inv2, &inv3};
    
    // Remove 30 ore, should remove 10 from each
    InventoryDelta consumed = Inventory::shared_update(inventories, 0, -30);
    
    std::cout << "\nAfter shared_update(-30):" << std::endl;
    std::cout << "consumed: " << consumed << std::endl;
    std::cout << "inv1: " << (int)inv1.amount(0) << std::endl;
    std::cout << "inv2: " << (int)inv2.amount(0) << std::endl;
    std::cout << "inv3: " << (int)inv3.amount(0) << std::endl;
    
    return 0;
}
