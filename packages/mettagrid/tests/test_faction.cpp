#include <cassert>
#include <iostream>
#include <memory>
#include <string>
#include <vector>

#include "core/grid_object.hpp"
#include "objects/alignable.hpp"
#include "objects/faction.hpp"
#include "objects/faction_config.hpp"
#include "objects/inventory_config.hpp"

// Test helper to create a basic Faction config
FactionConfig create_test_faction_config(const std::string& name, InventoryQuantity limit = 100) {
  FactionConfig config;
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

void test_faction_creation() {
  std::cout << "Testing Faction creation..." << std::endl;

  FactionConfig config = create_test_faction_config("test_faction");
  Faction faction(config);

  assert(faction.name == "test_faction");
  assert(faction.memberCount() == 0);

  std::cout << "✓ Faction creation test passed" << std::endl;
}

void test_faction_initial_inventory() {
  std::cout << "Testing Faction initial inventory..." << std::endl;

  FactionConfig config = create_test_faction_config("test_faction", 1000);
  config.initial_inventory[0] = 50;
  config.initial_inventory[1] = 25;

  Faction faction(config);

  assert(faction.inventory.amount(0) == 50);
  assert(faction.inventory.amount(1) == 25);

  std::cout << "✓ Faction initial inventory test passed" << std::endl;
}

void test_faction_add_member() {
  std::cout << "Testing Faction addMember..." << std::endl;

  FactionConfig config = create_test_faction_config("test_faction");
  Faction faction(config);

  TestGridObject obj1;
  TestGridObject obj2;

  faction.addMember(&obj1);
  assert(faction.memberCount() == 1);

  faction.addMember(&obj2);
  assert(faction.memberCount() == 2);

  // Adding same member again should not duplicate
  faction.addMember(&obj1);
  assert(faction.memberCount() == 2);

  std::cout << "✓ Faction addMember test passed" << std::endl;
}

void test_faction_remove_member() {
  std::cout << "Testing Faction removeMember..." << std::endl;

  FactionConfig config = create_test_faction_config("test_faction");
  Faction faction(config);

  TestGridObject obj1;
  TestGridObject obj2;

  faction.addMember(&obj1);
  faction.addMember(&obj2);
  assert(faction.memberCount() == 2);

  faction.removeMember(&obj1);
  assert(faction.memberCount() == 1);

  // Removing non-member should be safe
  faction.removeMember(&obj1);
  assert(faction.memberCount() == 1);

  faction.removeMember(&obj2);
  assert(faction.memberCount() == 0);

  std::cout << "✓ Faction removeMember test passed" << std::endl;
}

void test_alignable_set_faction() {
  std::cout << "Testing Alignable setFaction..." << std::endl;

  FactionConfig config = create_test_faction_config("test_faction");
  Faction faction(config);

  TestGridObject obj;

  // Initially no faction
  assert(obj.getFaction() == nullptr);
  assert(obj.faction_inventory() == nullptr);

  // Set faction
  obj.setFaction(&faction);
  assert(obj.getFaction() == &faction);
  assert(obj.faction_inventory() == &faction.inventory);
  assert(faction.memberCount() == 1);

  std::cout << "✓ Alignable setFaction test passed" << std::endl;
}

void test_alignable_clear_faction() {
  std::cout << "Testing Alignable clearFaction..." << std::endl;

  FactionConfig config = create_test_faction_config("test_faction");
  Faction faction(config);

  TestGridObject obj;
  obj.setFaction(&faction);
  assert(faction.memberCount() == 1);

  obj.clearFaction();
  assert(obj.getFaction() == nullptr);
  assert(obj.faction_inventory() == nullptr);
  assert(faction.memberCount() == 0);

  // Clearing again should be safe
  obj.clearFaction();
  assert(obj.getFaction() == nullptr);

  std::cout << "✓ Alignable clearFaction test passed" << std::endl;
}

void test_alignable_switch_faction() {
  std::cout << "Testing Alignable switching faction..." << std::endl;

  FactionConfig config1 = create_test_faction_config("faction1");
  FactionConfig config2 = create_test_faction_config("faction2");
  Faction faction1(config1);
  Faction faction2(config2);

  TestGridObject obj;
  obj.setFaction(&faction1);
  assert(faction1.memberCount() == 1);
  assert(faction2.memberCount() == 0);

  // Switch to different faction
  obj.setFaction(&faction2);
  assert(obj.getFaction() == &faction2);
  assert(faction1.memberCount() == 0);  // Removed from old
  assert(faction2.memberCount() == 1);  // Added to new

  std::cout << "✓ Alignable switching faction test passed" << std::endl;
}

void test_faction_inventory_access() {
  std::cout << "Testing faction inventory access..." << std::endl;

  FactionConfig config = create_test_faction_config("shared_storage", 1000);
  config.initial_inventory[0] = 100;
  Faction faction(config);

  TestGridObject obj;
  obj.setFaction(&faction);

  // Access faction inventory through grid object
  Inventory* inv = obj.faction_inventory();
  assert(inv != nullptr);
  assert(inv->amount(0) == 100);

  // Modify via faction_inventory
  inv->update(0, 50);
  assert(inv->amount(0) == 150);
  assert(faction.inventory.amount(0) == 150);

  std::cout << "✓ Faction inventory access test passed" << std::endl;
}

void test_multiple_objects_share_faction() {
  std::cout << "Testing multiple objects sharing faction inventory..." << std::endl;

  FactionConfig config = create_test_faction_config("shared_storage", 1000);
  config.initial_inventory[0] = 100;
  Faction faction(config);

  TestGridObject obj1;
  TestGridObject obj2;
  obj1.setFaction(&faction);
  obj2.setFaction(&faction);

  // Both objects should see the same inventory
  assert(obj1.faction_inventory()->amount(0) == 100);
  assert(obj2.faction_inventory()->amount(0) == 100);

  // Modification via one object is visible to the other
  obj1.faction_inventory()->update(0, 50);
  assert(obj2.faction_inventory()->amount(0) == 150);

  std::cout << "✓ Multiple objects sharing faction test passed" << std::endl;
}

int main() {
  std::cout << "Running Faction tests..." << std::endl;
  std::cout << "================================================" << std::endl;

  test_faction_creation();
  test_faction_initial_inventory();
  test_faction_add_member();
  test_faction_remove_member();
  test_alignable_set_faction();
  test_alignable_clear_faction();
  test_alignable_switch_faction();
  test_faction_inventory_access();
  test_multiple_objects_share_faction();

  std::cout << "================================================" << std::endl;
  std::cout << "All Faction tests passed! ✓" << std::endl;

  return 0;
}
