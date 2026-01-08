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

// Global resource names for tests
static std::vector<std::string> test_resource_names = {"gold", "silver"};

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
  TestGridObject(const std::string& type = "test_object") {
    this->type_name = type;
  }
};

void test_faction_creation() {
  std::cout << "Testing Faction creation..." << std::endl;

  FactionConfig config = create_test_faction_config("test_faction");
  Faction faction(config, &test_resource_names);

  assert(faction.name == "test_faction");
  assert(faction.memberCount() == 0);

  std::cout << "✓ Faction creation test passed" << std::endl;
}

void test_faction_initial_inventory() {
  std::cout << "Testing Faction initial inventory..." << std::endl;

  FactionConfig config = create_test_faction_config("test_faction", 1000);
  config.initial_inventory[0] = 50;
  config.initial_inventory[1] = 25;

  Faction faction(config, &test_resource_names);

  assert(faction.inventory.amount(0) == 50);
  assert(faction.inventory.amount(1) == 25);

  std::cout << "✓ Faction initial inventory test passed" << std::endl;
}

void test_faction_add_member() {
  std::cout << "Testing Faction addMember..." << std::endl;

  FactionConfig config = create_test_faction_config("test_faction");
  Faction faction(config, &test_resource_names);

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
  Faction faction(config, &test_resource_names);

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
  Faction faction(config, &test_resource_names);

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
  Faction faction(config, &test_resource_names);

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
  Faction faction1(config1, &test_resource_names);
  Faction faction2(config2, &test_resource_names);

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
  Faction faction(config, &test_resource_names);

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
  Faction faction(config, &test_resource_names);

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

void test_faction_aligned_counts() {
  std::cout << "Testing Faction aligned counts tracking..." << std::endl;

  FactionConfig config = create_test_faction_config("test_faction");
  Faction faction(config, &test_resource_names);

  // Create objects with different types
  TestGridObject charger1("charger");
  TestGridObject charger2("charger");
  TestGridObject extractor1("extractor");

  // Initially no aligned counts
  assert(faction.aligned_counts().empty());

  // Add objects via setFaction (which passes type_name)
  charger1.setFaction(&faction);
  assert(faction.aligned_counts().count("charger") == 1);
  assert(faction.aligned_counts().at("charger") == 1);
  assert(faction.stats.get("aligned.charger") == 1.0f);

  charger2.setFaction(&faction);
  assert(faction.aligned_counts().at("charger") == 2);
  assert(faction.stats.get("aligned.charger") == 2.0f);

  extractor1.setFaction(&faction);
  assert(faction.aligned_counts().at("extractor") == 1);
  assert(faction.stats.get("aligned.extractor") == 1.0f);

  // Remove a charger
  charger1.clearFaction();
  assert(faction.aligned_counts().at("charger") == 1);
  assert(faction.stats.get("aligned.charger") == 1.0f);

  std::cout << "✓ Faction aligned counts tracking test passed" << std::endl;
}

void test_faction_held_stats() {
  std::cout << "Testing Faction held duration stats..." << std::endl;

  FactionConfig config = create_test_faction_config("test_faction");
  Faction faction(config, &test_resource_names);

  TestGridObject charger1("charger");
  TestGridObject charger2("charger");

  charger1.setFaction(&faction);
  charger2.setFaction(&faction);

  // Initially no held stats
  assert(faction.stats.get("aligned.charger.held") == 0.0f);

  // Simulate one tick
  faction.update_held_stats();
  assert(faction.stats.get("aligned.charger.held") == 2.0f);

  // Simulate another tick
  faction.update_held_stats();
  assert(faction.stats.get("aligned.charger.held") == 4.0f);

  // Remove one charger and simulate tick
  charger1.clearFaction();
  faction.update_held_stats();
  assert(faction.stats.get("aligned.charger.held") == 5.0f);

  std::cout << "✓ Faction held duration stats test passed" << std::endl;
}

void test_faction_stats_tracker() {
  std::cout << "Testing Faction StatsTracker..." << std::endl;

  FactionConfig config = create_test_faction_config("test_faction");
  Faction faction(config, &test_resource_names);

  // Test basic stats operations
  faction.stats.incr("custom.stat");
  assert(faction.stats.get("custom.stat") == 1.0f);

  faction.stats.add("custom.stat", 5.0f);
  assert(faction.stats.get("custom.stat") == 6.0f);

  faction.stats.set("custom.stat", 10.0f);
  assert(faction.stats.get("custom.stat") == 10.0f);

  // Test to_dict
  auto stats_dict = faction.stats.to_dict();
  assert(stats_dict.count("custom.stat") == 1);
  assert(stats_dict.at("custom.stat") == 10.0f);

  std::cout << "✓ Faction StatsTracker test passed" << std::endl;
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
  test_faction_aligned_counts();
  test_faction_held_stats();
  test_faction_stats_tracker();

  std::cout << "================================================" << std::endl;
  std::cout << "All Faction tests passed! ✓" << std::endl;

  return 0;
}
