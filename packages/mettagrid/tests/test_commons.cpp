#include <cassert>
#include <iostream>
#include <memory>
#include <string>
#include <vector>

#include "core/grid_object.hpp"
#include "objects/commons.hpp"
#include "objects/commons_config.hpp"
#include "objects/inventory_config.hpp"

// Test helper to create a basic Commons config
CommonsConfig create_test_commons_config(const std::string& name, InventoryQuantity limit = 100) {
  CommonsConfig config;
  config.name = name;
  config.inventory_config.limit_defs.push_back(LimitDef({0}, limit));  // Resource type 0
  config.inventory_config.limit_defs.push_back(LimitDef({1}, limit));  // Resource type 1
  return config;
}

// Simple GridObject subclass for testing
class TestGridObject : public GridObject {
public:
  TestGridObject() = default;
};

void test_commons_creation() {
  std::cout << "Testing Commons creation..." << std::endl;

  CommonsConfig config = create_test_commons_config("test_commons");
  Commons commons(config);

  assert(commons.name == "test_commons");
  assert(commons.memberCount() == 0);

  std::cout << "✓ Commons creation test passed" << std::endl;
}

void test_commons_initial_inventory() {
  std::cout << "Testing Commons initial inventory..." << std::endl;

  CommonsConfig config = create_test_commons_config("test_commons", 1000);
  config.initial_inventory[0] = 50;
  config.initial_inventory[1] = 25;

  Commons commons(config);

  assert(commons.inventory.amount(0) == 50);
  assert(commons.inventory.amount(1) == 25);

  std::cout << "✓ Commons initial inventory test passed" << std::endl;
}

void test_commons_add_member() {
  std::cout << "Testing Commons addMember..." << std::endl;

  CommonsConfig config = create_test_commons_config("test_commons");
  Commons commons(config);

  TestGridObject obj1;
  TestGridObject obj2;

  commons.addMember(&obj1);
  assert(commons.memberCount() == 1);

  commons.addMember(&obj2);
  assert(commons.memberCount() == 2);

  // Adding same member again should not duplicate
  commons.addMember(&obj1);
  assert(commons.memberCount() == 2);

  std::cout << "✓ Commons addMember test passed" << std::endl;
}

void test_commons_remove_member() {
  std::cout << "Testing Commons removeMember..." << std::endl;

  CommonsConfig config = create_test_commons_config("test_commons");
  Commons commons(config);

  TestGridObject obj1;
  TestGridObject obj2;

  commons.addMember(&obj1);
  commons.addMember(&obj2);
  assert(commons.memberCount() == 2);

  commons.removeMember(&obj1);
  assert(commons.memberCount() == 1);

  // Removing non-member should be safe
  commons.removeMember(&obj1);
  assert(commons.memberCount() == 1);

  commons.removeMember(&obj2);
  assert(commons.memberCount() == 0);

  std::cout << "✓ Commons removeMember test passed" << std::endl;
}

void test_gridobject_set_commons() {
  std::cout << "Testing GridObject setCommons..." << std::endl;

  CommonsConfig config = create_test_commons_config("test_commons");
  Commons commons(config);

  TestGridObject obj;

  // Initially no commons
  assert(obj.getCommons() == nullptr);
  assert(obj.commons_inventory() == nullptr);

  // Set commons
  obj.setCommons(&commons);
  assert(obj.getCommons() == &commons);
  assert(obj.commons_inventory() == &commons.inventory);
  assert(commons.memberCount() == 1);

  std::cout << "✓ GridObject setCommons test passed" << std::endl;
}

void test_gridobject_clear_commons() {
  std::cout << "Testing GridObject clearCommons..." << std::endl;

  CommonsConfig config = create_test_commons_config("test_commons");
  Commons commons(config);

  TestGridObject obj;
  obj.setCommons(&commons);
  assert(commons.memberCount() == 1);

  obj.clearCommons();
  assert(obj.getCommons() == nullptr);
  assert(obj.commons_inventory() == nullptr);
  assert(commons.memberCount() == 0);

  // Clearing again should be safe
  obj.clearCommons();
  assert(obj.getCommons() == nullptr);

  std::cout << "✓ GridObject clearCommons test passed" << std::endl;
}

void test_gridobject_switch_commons() {
  std::cout << "Testing GridObject switching commons..." << std::endl;

  CommonsConfig config1 = create_test_commons_config("commons1");
  CommonsConfig config2 = create_test_commons_config("commons2");
  Commons commons1(config1);
  Commons commons2(config2);

  TestGridObject obj;
  obj.setCommons(&commons1);
  assert(commons1.memberCount() == 1);
  assert(commons2.memberCount() == 0);

  // Switch to different commons
  obj.setCommons(&commons2);
  assert(obj.getCommons() == &commons2);
  assert(commons1.memberCount() == 0);  // Removed from old
  assert(commons2.memberCount() == 1);  // Added to new

  std::cout << "✓ GridObject switching commons test passed" << std::endl;
}

void test_commons_inventory_access() {
  std::cout << "Testing commons inventory access..." << std::endl;

  CommonsConfig config = create_test_commons_config("shared_storage", 1000);
  config.initial_inventory[0] = 100;
  Commons commons(config);

  TestGridObject obj;
  obj.setCommons(&commons);

  // Access commons inventory through grid object
  Inventory* inv = obj.commons_inventory();
  assert(inv != nullptr);
  assert(inv->amount(0) == 100);

  // Modify via commons_inventory
  inv->update(0, 50);
  assert(inv->amount(0) == 150);
  assert(commons.inventory.amount(0) == 150);

  std::cout << "✓ Commons inventory access test passed" << std::endl;
}

void test_multiple_objects_share_commons() {
  std::cout << "Testing multiple objects sharing commons inventory..." << std::endl;

  CommonsConfig config = create_test_commons_config("shared_storage", 1000);
  config.initial_inventory[0] = 100;
  Commons commons(config);

  TestGridObject obj1;
  TestGridObject obj2;
  obj1.setCommons(&commons);
  obj2.setCommons(&commons);

  // Both objects should see the same inventory
  assert(obj1.commons_inventory()->amount(0) == 100);
  assert(obj2.commons_inventory()->amount(0) == 100);

  // Modification via one object is visible to the other
  obj1.commons_inventory()->update(0, 50);
  assert(obj2.commons_inventory()->amount(0) == 150);

  std::cout << "✓ Multiple objects sharing commons test passed" << std::endl;
}

int main() {
  std::cout << "Running Commons tests..." << std::endl;
  std::cout << "================================================" << std::endl;

  test_commons_creation();
  test_commons_initial_inventory();
  test_commons_add_member();
  test_commons_remove_member();
  test_gridobject_set_commons();
  test_gridobject_clear_commons();
  test_gridobject_switch_commons();
  test_commons_inventory_access();
  test_multiple_objects_share_commons();

  std::cout << "================================================" << std::endl;
  std::cout << "All Commons tests passed! ✓" << std::endl;

  return 0;
}
