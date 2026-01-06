#include <cassert>
#include <iostream>

#include "objects/has_inventory.hpp"
#include "objects/inventory_config.hpp"

// Test helper function to create an inventory config with specified limits
InventoryConfig create_test_inventory_config(InventoryQuantity limit) {
  InventoryConfig config;
  // Set individual limits for resource types 0, 1, and 2
  config.limit_defs.push_back(LimitDef({0}, limit));  // Resource type 0
  config.limit_defs.push_back(LimitDef({1}, limit));  // Resource type 1
  config.limit_defs.push_back(LimitDef({2}, limit));  // Resource type 2
  return config;
}

void test_basic_transfer() {
  std::cout << "Testing basic transfer..." << std::endl;

  // Create two inventories with capacity 100 each
  InventoryConfig config = create_test_inventory_config(100);
  HasInventory source(config);
  HasInventory target(config);

  // Give source 50 units of resource 0
  source.inventory.update(0, 50);

  // Transfer 30 units from source to target
  InventoryDelta transferred = HasInventory::transfer_resources(source.inventory, target.inventory, 0, 30, false);

  // Check results
  assert(transferred == 30);
  assert(source.inventory.amount(0) == 20);  // 50 - 30
  assert(target.inventory.amount(0) == 30);  // 0 + 30

  std::cout << "✓ Basic transfer test passed" << std::endl;
}

void test_transfer_with_limited_source() {
  std::cout << "Testing transfer with limited source..." << std::endl;

  InventoryConfig config = create_test_inventory_config(100);
  HasInventory source(config);
  HasInventory target(config);

  // Give source only 20 units
  source.inventory.update(0, 20);

  // Try to transfer 30 units (but source only has 20)
  InventoryDelta transferred = HasInventory::transfer_resources(source.inventory, target.inventory, 0, 30, false);

  // Should only transfer 20
  assert(transferred == 20);
  assert(source.inventory.amount(0) == 0);   // All gone
  assert(target.inventory.amount(0) == 20);  // Received 20

  std::cout << "✓ Limited source test passed" << std::endl;
}

void test_transfer_with_limited_target() {
  std::cout << "Testing transfer with limited target..." << std::endl;

  // Source has large capacity, target has small capacity
  InventoryConfig source_config = create_test_inventory_config(100);
  InventoryConfig target_config = create_test_inventory_config(25);
  HasInventory source(source_config);
  HasInventory target(target_config);

  // Give source 50 units
  source.inventory.update(0, 50);

  // Give target 10 units (so it can only accept 15 more)
  target.inventory.update(0, 10);

  // Try to transfer 30 units (but target can only accept 15)
  InventoryDelta transferred = HasInventory::transfer_resources(source.inventory, target.inventory, 0, 30, false);

  // Should only transfer 15
  assert(transferred == 15);
  assert(source.inventory.amount(0) == 35);  // 50 - 15
  assert(target.inventory.amount(0) == 25);  // 10 + 15 (at max)

  std::cout << "✓ Limited target test passed" << std::endl;
}

void test_destroy_untransferred_resources_true() {
  std::cout << "Testing destroy_untransferred_resources=true..." << std::endl;

  // Source has large capacity, target has small capacity
  InventoryConfig source_config = create_test_inventory_config(100);
  InventoryConfig target_config = create_test_inventory_config(20);
  HasInventory source(source_config);
  HasInventory target(target_config);

  // Give source 50 units
  source.inventory.update(0, 50);

  // Give target 15 units (so it can only accept 5 more)
  target.inventory.update(0, 15);

  // Try to transfer 30 units with destroy_untransferred_resources=true
  // Source should lose 30, but target can only receive 5
  InventoryDelta transferred = HasInventory::transfer_resources(source.inventory, target.inventory, 0, 30, true);

  // Should transfer 5, but source loses 30
  assert(transferred == 5);
  assert(source.inventory.amount(0) == 20);  // 50 - 30 (lost 30)
  assert(target.inventory.amount(0) == 20);  // 15 + 5 (at max, 25 destroyed)

  std::cout << "✓ Destroy untransferred resources test passed" << std::endl;
}

void test_destroy_untransferred_with_limited_source() {
  std::cout << "Testing destroy_untransferred with limited source..." << std::endl;

  InventoryConfig source_config = create_test_inventory_config(100);
  InventoryConfig target_config = create_test_inventory_config(20);
  HasInventory source(source_config);
  HasInventory target(target_config);

  // Give source only 15 units
  source.inventory.update(0, 15);

  // Give target 10 units (can accept 10 more)
  target.inventory.update(0, 10);

  // Try to transfer 25 units with destroy_untransferred_resources=true
  // Source can only give 15, target can accept 10
  InventoryDelta transferred = HasInventory::transfer_resources(source.inventory, target.inventory, 0, 25, true);

  // Should transfer 10, source loses all 15
  assert(transferred == 10);
  assert(source.inventory.amount(0) == 0);   // Lost all 15
  assert(target.inventory.amount(0) == 20);  // 10 + 10 (5 destroyed)

  std::cout << "✓ Destroy untransferred with limited source test passed" << std::endl;
}

void test_zero_delta() {
  std::cout << "Testing zero delta..." << std::endl;

  InventoryConfig config = create_test_inventory_config(100);
  HasInventory source(config);
  HasInventory target(config);

  source.inventory.update(0, 50);
  target.inventory.update(0, 20);

  // Transfer with delta=0 should do nothing
  InventoryDelta transferred = HasInventory::transfer_resources(source.inventory, target.inventory, 0, 0, false);

  assert(transferred == 0);
  assert(source.inventory.amount(0) == 50);  // Unchanged
  assert(target.inventory.amount(0) == 20);  // Unchanged

  std::cout << "✓ Zero delta test passed" << std::endl;
}

void test_negative_delta() {
  std::cout << "Testing negative delta..." << std::endl;

  InventoryConfig config = create_test_inventory_config(100);
  HasInventory source(config);
  HasInventory target(config);

  source.inventory.update(0, 50);
  target.inventory.update(0, 20);

  // Transfer with negative delta should do nothing
  InventoryDelta transferred = HasInventory::transfer_resources(source.inventory, target.inventory, 0, -10, false);

  assert(transferred == 0);
  assert(source.inventory.amount(0) == 50);  // Unchanged
  assert(target.inventory.amount(0) == 20);  // Unchanged

  std::cout << "✓ Negative delta test passed" << std::endl;
}

void test_transfer_to_full_inventory() {
  std::cout << "Testing transfer to full inventory..." << std::endl;

  InventoryConfig config = create_test_inventory_config(50);
  HasInventory source(config);
  HasInventory target(config);

  source.inventory.update(0, 30);
  target.inventory.update(0, 50);  // Target is full

  // Try to transfer to full inventory
  InventoryDelta transferred = HasInventory::transfer_resources(source.inventory, target.inventory, 0, 10, false);

  assert(transferred == 0);
  assert(source.inventory.amount(0) == 30);  // Unchanged
  assert(target.inventory.amount(0) == 50);  // Still full

  std::cout << "✓ Transfer to full inventory test passed" << std::endl;
}

void test_transfer_from_empty_inventory() {
  std::cout << "Testing transfer from empty inventory..." << std::endl;

  InventoryConfig config = create_test_inventory_config(100);
  HasInventory source(config);
  HasInventory target(config);

  // Source is empty
  target.inventory.update(0, 20);

  // Try to transfer from empty inventory
  InventoryDelta transferred = HasInventory::transfer_resources(source.inventory, target.inventory, 0, 10, false);

  assert(transferred == 0);
  assert(source.inventory.amount(0) == 0);   // Still empty
  assert(target.inventory.amount(0) == 20);  // Unchanged

  std::cout << "✓ Transfer from empty inventory test passed" << std::endl;
}

// ============================================================
// Dynamic Inventory Limits with Modifiers Tests
// ============================================================

void test_limit_defs_basic() {
  std::cout << "Testing limit_defs basic functionality..." << std::endl;

  // Create inventory using limit_defs format
  InventoryConfig config = create_test_inventory_config(100);
  HasInventory inv(config);

  // Add 50 units of resource 0
  InventoryDelta added = inv.inventory.update(0, 50);
  assert(added == 50);
  assert(inv.inventory.amount(0) == 50);

  // Try to add 60 more (should be clamped to 50 more to reach limit of 100)
  InventoryDelta added2 = inv.inventory.update(0, 60);
  assert(added2 == 50);
  assert(inv.inventory.amount(0) == 100);

  std::cout << "✓ limit_defs basic test passed" << std::endl;
}

void test_dynamic_limit_with_modifier() {
  std::cout << "Testing dynamic limit with modifier..." << std::endl;

  // Example: battery limit starts at 0, each gear adds +1 battery capacity
  // Resource 0 = gear, Resource 1 = battery
  InventoryConfig config;
  config.limit_defs.push_back(LimitDef({0}, 10));  // gear has fixed limit of 10
  // battery has base limit of 0, but each gear adds 5 capacity
  config.limit_defs.push_back(LimitDef({1}, 0, {{0, 5}}));

  HasInventory inv(config);

  // Initially, we can't add any batteries (limit is 0)
  InventoryDelta added = inv.inventory.update(1, 10);
  assert(added == 0);
  assert(inv.inventory.amount(1) == 0);

  // Add 2 gears - this should increase battery limit to 10 (2 * 5)
  inv.inventory.update(0, 2);
  assert(inv.inventory.amount(0) == 2);

  // Now we should be able to add up to 10 batteries
  InventoryDelta added2 = inv.inventory.update(1, 10);
  assert(added2 == 10);
  assert(inv.inventory.amount(1) == 10);

  // Try to add more batteries (should fail, at limit)
  InventoryDelta added3 = inv.inventory.update(1, 5);
  assert(added3 == 0);

  // Add 1 more gear - battery limit should increase to 15
  inv.inventory.update(0, 1);
  assert(inv.inventory.amount(0) == 3);

  // Now we should be able to add 5 more batteries
  InventoryDelta added4 = inv.inventory.update(1, 5);
  assert(added4 == 5);
  assert(inv.inventory.amount(1) == 15);

  std::cout << "✓ Dynamic limit with modifier test passed" << std::endl;
}

void test_dynamic_limit_chain() {
  std::cout << "Testing dynamic limit chain (gear -> battery -> energy)..." << std::endl;

  // Example: gear has fixed limit, battery depends on gear, energy depends on battery
  // Resource 0 = gear, Resource 1 = battery, Resource 2 = energy
  InventoryConfig config;
  config.limit_defs.push_back(LimitDef({0}, 5));             // gear has fixed limit of 5
  config.limit_defs.push_back(LimitDef({1}, 0, {{0, 1}}));   // each gear adds +1 battery capacity
  config.limit_defs.push_back(LimitDef({2}, 0, {{1, 25}}));  // each battery adds +25 energy capacity

  HasInventory inv(config);

  // Initially can't add batteries or energy
  assert(inv.inventory.update(1, 5) == 0);
  assert(inv.inventory.update(2, 100) == 0);

  // Add 3 gears
  assert(inv.inventory.update(0, 3) == 3);

  // Now can add up to 3 batteries
  assert(inv.inventory.update(1, 3) == 3);
  assert(inv.inventory.amount(1) == 3);

  // Now can add up to 75 energy (3 batteries * 25)
  assert(inv.inventory.update(2, 75) == 75);
  assert(inv.inventory.amount(2) == 75);

  // Can't add more energy (at limit)
  assert(inv.inventory.update(2, 10) == 0);

  // Add 1 more battery (total 4, but we only have 3 gear capacity)
  // This should fail since battery limit is 3
  assert(inv.inventory.update(1, 1) == 0);

  std::cout << "✓ Dynamic limit chain test passed" << std::endl;
}

void test_free_space_with_modifiers() {
  std::cout << "Testing free_space with modifiers..." << std::endl;

  // Resource 0 = gear, Resource 1 = battery
  InventoryConfig config;
  config.limit_defs.push_back(LimitDef({0}, 10));           // gear limit 10
  config.limit_defs.push_back(LimitDef({1}, 0, {{0, 5}}));  // battery limit depends on gear

  HasInventory inv(config);

  // No gears, so battery free_space should be 0
  assert(inv.inventory.free_space(1) == 0);

  // Add 2 gears, battery free_space should be 10
  inv.inventory.update(0, 2);
  assert(inv.inventory.free_space(1) == 10);

  // Add 3 batteries, free_space should be 7
  inv.inventory.update(1, 3);
  assert(inv.inventory.free_space(1) == 7);

  // Add 1 more gear, free_space should be 12 (15 limit - 3 used)
  inv.inventory.update(0, 1);
  assert(inv.inventory.free_space(1) == 12);

  std::cout << "✓ free_space with modifiers test passed" << std::endl;
}

void test_limit_reduces_when_modifier_removed() {
  std::cout << "Testing limit reduction when modifier item is removed..." << std::endl;

  // Resource 0 = gear, Resource 1 = battery
  InventoryConfig config;
  config.limit_defs.push_back(LimitDef({0}, 10));           // gear limit 10
  config.limit_defs.push_back(LimitDef({1}, 0, {{0, 5}}));  // battery limit depends on gear

  HasInventory inv(config);

  // Add 4 gears (battery limit = 20)
  inv.inventory.update(0, 4);
  // Add 18 batteries
  inv.inventory.update(1, 18);
  assert(inv.inventory.amount(1) == 18);

  // Remove 2 gears (battery limit now = 10)
  inv.inventory.update(0, -2);
  assert(inv.inventory.amount(0) == 2);

  // Batteries should now be 10, since the limit is automatically enforced.
  assert(inv.inventory.amount(1) == 10);

  // Free_space should be 0 (over limit)
  assert(inv.inventory.free_space(1) == 0);

  std::cout << "✓ Limit reduction when modifier removed test passed" << std::endl;
}

void test_recursive_limit_reduction_when_dropping_inventory() {
  std::cout << "Testing recursive limit reduction when dropping inventory causes limit changes..." << std::endl;

  // Resource 0 = gear, Resource 1 = battery, Resource 2 = energy
  // Chain: gear -> battery -> energy
  InventoryConfig config;
  config.limit_defs.push_back(LimitDef({0}, 10));            // gear has fixed limit of 10
  config.limit_defs.push_back(LimitDef({1}, 0, {{0, 2}}));   // each gear adds +2 battery capacity
  config.limit_defs.push_back(LimitDef({2}, 0, {{1, 10}}));  // each battery adds +10 energy capacity

  HasInventory inv(config);

  // Add 5 gears (battery limit = 10)
  inv.inventory.update(0, 5);
  assert(inv.inventory.amount(0) == 5);

  // Add 10 batteries (at limit)
  inv.inventory.update(1, 10);
  assert(inv.inventory.amount(1) == 10);

  // Add 100 energy (at limit: 10 batteries * 10 = 100)
  inv.inventory.update(2, 100);
  assert(inv.inventory.amount(2) == 100);

  // Now remove 3 gears (from 5 to 2)
  // This should trigger recursive limit enforcement:
  // 1. Battery limit reduces from 10 to 4 (2 gears * 2)
  // 2. Batteries drop from 10 to 4
  // 3. Energy limit reduces from 100 to 40 (4 batteries * 10)
  // 4. Energy drops from 100 to 40
  inv.inventory.update(0, -3);
  assert(inv.inventory.amount(0) == 2);

  // Batteries should be reduced to 4 (new limit)
  assert(inv.inventory.amount(1) == 4);

  // Energy should be reduced to 40 (new limit: 4 batteries * 10)
  assert(inv.inventory.amount(2) == 40);

  // Verify free_space is 0 for all (at limits)
  assert(inv.inventory.free_space(0) == 8);  // 10 - 2 = 8
  assert(inv.inventory.free_space(1) == 0);  // At limit
  assert(inv.inventory.free_space(2) == 0);  // At limit

  std::cout << "✓ Recursive limit reduction when dropping inventory test passed" << std::endl;
}

void test_transfer_with_modifier_limits() {
  std::cout << "Testing transfer with modifier-based limits..." << std::endl;

  // Resource 0 = gear, Resource 1 = battery
  InventoryConfig source_config;
  source_config.limit_defs.push_back(LimitDef({0}, 10));
  source_config.limit_defs.push_back(LimitDef({1}, 100));  // source has fixed battery limit

  InventoryConfig target_config;
  target_config.limit_defs.push_back(LimitDef({0}, 10));
  target_config.limit_defs.push_back(LimitDef({1}, 0, {{0, 5}}));  // target battery depends on gear

  HasInventory source(source_config);
  HasInventory target(target_config);

  // Source has lots of batteries
  source.inventory.update(1, 50);

  // Target has 2 gears (battery limit = 10)
  target.inventory.update(0, 2);

  // Try to transfer 20 batteries - should only transfer 10
  InventoryDelta transferred = HasInventory::transfer_resources(source.inventory, target.inventory, 1, 20, false);

  assert(transferred == 10);
  assert(source.inventory.amount(1) == 40);
  assert(target.inventory.amount(1) == 10);

  std::cout << "✓ Transfer with modifier limits test passed" << std::endl;
}

int main() {
  std::cout << "Running HasInventory::transfer_resources tests..." << std::endl;
  std::cout << "================================================" << std::endl;

  test_basic_transfer();
  test_transfer_with_limited_source();
  test_transfer_with_limited_target();
  test_destroy_untransferred_resources_true();
  test_destroy_untransferred_with_limited_source();
  test_zero_delta();
  test_negative_delta();
  test_transfer_to_full_inventory();
  test_transfer_from_empty_inventory();

  std::cout << "================================================" << std::endl;
  std::cout << "Running Dynamic Inventory Limits tests..." << std::endl;
  std::cout << "================================================" << std::endl;

  test_limit_defs_basic();
  test_dynamic_limit_with_modifier();
  test_dynamic_limit_chain();
  test_free_space_with_modifiers();
  test_limit_reduces_when_modifier_removed();
  test_recursive_limit_reduction_when_dropping_inventory();
  test_transfer_with_modifier_limits();

  std::cout << "================================================" << std::endl;
  std::cout << "All tests passed! ✓" << std::endl;

  return 0;
}
