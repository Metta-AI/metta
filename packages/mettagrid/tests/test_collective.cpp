#include <cassert>
#include <iostream>
#include <memory>
#include <string>
#include <vector>

#include "core/grid_object.hpp"
#include "objects/alignable.hpp"
#include "objects/collective.hpp"
#include "objects/collective_config.hpp"
#include "objects/inventory_config.hpp"

// Test helper to create a basic Collective config
CollectiveConfig create_test_collective_config(const std::string& name, InventoryQuantity limit = 100) {
  CollectiveConfig config;
  config.name = name;
  config.inventory_config.limit_defs.push_back(LimitDef({0}, limit));  // Resource type 0
  config.inventory_config.limit_defs.push_back(LimitDef({1}, limit));  // Resource type 1
  return config;
}

// Simple GridObject subclass for testing that is also Alignable
class TestGridObject : public GridObject, public Alignable {
public:
  TestGridObject() = default;
};

void test_collective_creation() {
  std::cout << "Testing Collective creation..." << std::endl;

  CollectiveConfig config = create_test_collective_config("test_collective");
  Collective collective(config);

  assert(collective.name == "test_collective");
  assert(collective.memberCount() == 0);

  std::cout << "✓ Collective creation test passed" << std::endl;
}

void test_collective_initial_inventory() {
  std::cout << "Testing Collective initial inventory..." << std::endl;

  CollectiveConfig config = create_test_collective_config("test_collective", 1000);
  config.initial_inventory[0] = 50;
  config.initial_inventory[1] = 25;

  Collective collective(config);

  assert(collective.inventory.amount(0) == 50);
  assert(collective.inventory.amount(1) == 25);

  std::cout << "✓ Collective initial inventory test passed" << std::endl;
}

void test_collective_add_member() {
  std::cout << "Testing Collective addMember..." << std::endl;

  CollectiveConfig config = create_test_collective_config("test_collective");
  Collective collective(config);

  TestGridObject obj1;
  TestGridObject obj2;

  collective.addMember(&obj1);
  assert(collective.memberCount() == 1);

  collective.addMember(&obj2);
  assert(collective.memberCount() == 2);

  // Adding same member again should not duplicate
  collective.addMember(&obj1);
  assert(collective.memberCount() == 2);

  std::cout << "✓ Collective addMember test passed" << std::endl;
}

void test_collective_remove_member() {
  std::cout << "Testing Collective removeMember..." << std::endl;

  CollectiveConfig config = create_test_collective_config("test_collective");
  Collective collective(config);

  TestGridObject obj1;
  TestGridObject obj2;

  collective.addMember(&obj1);
  collective.addMember(&obj2);
  assert(collective.memberCount() == 2);

  collective.removeMember(&obj1);
  assert(collective.memberCount() == 1);

  // Removing non-member should be safe
  collective.removeMember(&obj1);
  assert(collective.memberCount() == 1);

  collective.removeMember(&obj2);
  assert(collective.memberCount() == 0);

  std::cout << "✓ Collective removeMember test passed" << std::endl;
}

void test_alignable_set_collective() {
  std::cout << "Testing Alignable setCollective..." << std::endl;

  CollectiveConfig config = create_test_collective_config("test_collective");
  Collective collective(config);

  TestGridObject obj;

  // Initially no collective
  assert(obj.getCollective() == nullptr);
  assert(obj.collective_inventory() == nullptr);

  // Set collective
  obj.setCollective(&collective);
  assert(obj.getCollective() == &collective);
  assert(obj.collective_inventory() == &collective.inventory);
  assert(collective.memberCount() == 1);

  std::cout << "✓ Alignable setCollective test passed" << std::endl;
}

void test_alignable_clear_collective() {
  std::cout << "Testing Alignable clearCollective..." << std::endl;

  CollectiveConfig config = create_test_collective_config("test_collective");
  Collective collective(config);

  TestGridObject obj;
  obj.setCollective(&collective);
  assert(collective.memberCount() == 1);

  obj.clearCollective();
  assert(obj.getCollective() == nullptr);
  assert(obj.collective_inventory() == nullptr);
  assert(collective.memberCount() == 0);

  // Clearing again should be safe
  obj.clearCollective();
  assert(obj.getCollective() == nullptr);

  std::cout << "✓ Alignable clearCollective test passed" << std::endl;
}

void test_alignable_switch_collective() {
  std::cout << "Testing Alignable switching collective..." << std::endl;

  CollectiveConfig config1 = create_test_collective_config("collective1");
  CollectiveConfig config2 = create_test_collective_config("collective2");
  Collective collective1(config1);
  Collective collective2(config2);

  TestGridObject obj;
  obj.setCollective(&collective1);
  assert(collective1.memberCount() == 1);
  assert(collective2.memberCount() == 0);

  // Switch to different collective
  obj.setCollective(&collective2);
  assert(obj.getCollective() == &collective2);
  assert(collective1.memberCount() == 0);  // Removed from old
  assert(collective2.memberCount() == 1);  // Added to new

  std::cout << "✓ Alignable switching collective test passed" << std::endl;
}

void test_collective_inventory_access() {
  std::cout << "Testing collective inventory access..." << std::endl;

  CollectiveConfig config = create_test_collective_config("shared_storage", 1000);
  config.initial_inventory[0] = 100;
  Collective collective(config);

  TestGridObject obj;
  obj.setCollective(&collective);

  // Access collective inventory through grid object
  Inventory* inv = obj.collective_inventory();
  assert(inv != nullptr);
  assert(inv->amount(0) == 100);

  // Modify via collective_inventory
  inv->update(0, 50);
  assert(inv->amount(0) == 150);
  assert(collective.inventory.amount(0) == 150);

  std::cout << "✓ Collective inventory access test passed" << std::endl;
}

void test_multiple_objects_share_collective() {
  std::cout << "Testing multiple objects sharing collective inventory..." << std::endl;

  CollectiveConfig config = create_test_collective_config("shared_storage", 1000);
  config.initial_inventory[0] = 100;
  Collective collective(config);

  TestGridObject obj1;
  TestGridObject obj2;
  obj1.setCollective(&collective);
  obj2.setCollective(&collective);

  // Both objects should see the same inventory
  assert(obj1.collective_inventory()->amount(0) == 100);
  assert(obj2.collective_inventory()->amount(0) == 100);

  // Modification via one object is visible to the other
  obj1.collective_inventory()->update(0, 50);
  assert(obj2.collective_inventory()->amount(0) == 150);

  std::cout << "✓ Multiple objects sharing collective test passed" << std::endl;
}

int main() {
  std::cout << "Running Collective tests..." << std::endl;
  std::cout << "================================================" << std::endl;

  test_collective_creation();
  test_collective_initial_inventory();
  test_collective_add_member();
  test_collective_remove_member();
  test_alignable_set_collective();
  test_alignable_clear_collective();
  test_alignable_switch_collective();
  test_collective_inventory_access();
  test_multiple_objects_share_collective();

  std::cout << "================================================" << std::endl;
  std::cout << "All Collective tests passed! ✓" << std::endl;

  return 0;
}
