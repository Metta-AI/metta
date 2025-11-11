#include <cassert>
#include <iostream>

#include "objects/has_inventory.hpp"
#include "objects/inventory_config.hpp"

// Test helper function to create an inventory config with specified limits
InventoryConfig create_test_inventory_config(InventoryQuantity limit) {
  InventoryConfig config;
  // Set individual limits for resource types 0, 1, and 2
  config.limits.push_back({{0}, limit});  // Resource type 0
  config.limits.push_back({{1}, limit});  // Resource type 1
  config.limits.push_back({{2}, limit});  // Resource type 2
  return config;
}

void test_basic_transfer() {
  std::cout << "Testing basic transfer..." << std::endl;

  // Create two inventories with capacity 100 each
  InventoryConfig config = create_test_inventory_config(100);
  HasInventory source(config);
  HasInventory target(config);

  // Give source 50 units of resource 0
  source.update_inventory(0, 50);

  // Transfer 30 units from source to target
  InventoryDelta transferred = HasInventory::transfer_resources(source, target, 0, 30);

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
  source.update_inventory(0, 20);

  // Try to transfer 30 units (but source only has 20)
  InventoryDelta transferred = HasInventory::transfer_resources(source, target, 0, 30);

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
  source.update_inventory(0, 50);

  // Give target 10 units (so it can only accept 15 more)
  target.update_inventory(0, 10);

  // Try to transfer 30 units (but target can only accept 15)
  InventoryDelta transferred = HasInventory::transfer_resources(source, target, 0, 30);

  // Should only transfer 15
  assert(transferred == 15);
  assert(source.inventory.amount(0) == 35);  // 50 - 15
  assert(target.inventory.amount(0) == 25);  // 10 + 15 (at max)

  std::cout << "✓ Limited target test passed" << std::endl;
}

void test_zero_delta() {
  std::cout << "Testing zero delta..." << std::endl;

  InventoryConfig config = create_test_inventory_config(100);
  HasInventory source(config);
  HasInventory target(config);

  source.update_inventory(0, 50);
  target.update_inventory(0, 20);

  // Transfer with delta=0 should do nothing
  InventoryDelta transferred = HasInventory::transfer_resources(source, target, 0, 0);

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

  source.update_inventory(0, 50);
  target.update_inventory(0, 20);

  // Transfer with negative delta should do nothing
  InventoryDelta transferred = HasInventory::transfer_resources(source, target, 0, -10);

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

  source.update_inventory(0, 30);
  target.update_inventory(0, 50);  // Target is full

  // Try to transfer to full inventory
  InventoryDelta transferred = HasInventory::transfer_resources(source, target, 0, 10);

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
  target.update_inventory(0, 20);

  // Try to transfer from empty inventory
  InventoryDelta transferred = HasInventory::transfer_resources(source, target, 0, 10);

  assert(transferred == 0);
  assert(source.inventory.amount(0) == 0);   // Still empty
  assert(target.inventory.amount(0) == 20);  // Unchanged

  std::cout << "✓ Transfer from empty inventory test passed" << std::endl;
}

int main() {
  std::cout << "Running HasInventory::transfer_resources tests..." << std::endl;
  std::cout << "================================================" << std::endl;

  test_basic_transfer();
  test_transfer_with_limited_source();
  test_transfer_with_limited_target();
  test_zero_delta();
  test_negative_delta();
  test_transfer_to_full_inventory();
  test_transfer_from_empty_inventory();

  std::cout << "================================================" << std::endl;
  std::cout << "All tests passed! ✓" << std::endl;

  return 0;
}
