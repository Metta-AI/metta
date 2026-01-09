#include <cassert>
#include <iostream>
#include <memory>
#include <string>
#include <vector>

#include "actions/activation_context.hpp"
#include "actions/activation_handler.hpp"
#include "actions/activation_handler_config.hpp"
#include "actions/filters/filter.hpp"
#include "actions/mutations/mutation.hpp"
#include "core/grid_object.hpp"
#include "objects/alignable.hpp"
#include "objects/collective.hpp"
#include "objects/collective_config.hpp"
#include "objects/has_inventory.hpp"
#include "objects/inventory_config.hpp"

using namespace mettagrid;

// Resource names for testing
static std::vector<std::string> test_resource_names = {"health", "energy", "gold"};

// Simple GridObject subclass that has an inventory and is alignable
class TestActivationObject : public GridObject, public HasInventory, public Alignable {
public:
  explicit TestActivationObject(const std::string& type = "test_object", ObservationType initial_vibe = 0)
      : HasInventory(create_inventory_config()) {
    type_name = type;
    vibe = initial_vibe;
    location.r = 0;
    location.c = 0;
  }

  static InventoryConfig create_inventory_config() {
    InventoryConfig config;
    config.limit_defs.push_back(LimitDef({0}, 1000));  // health
    config.limit_defs.push_back(LimitDef({1}, 1000));  // energy
    config.limit_defs.push_back(LimitDef({2}, 1000));  // gold
    return config;
  }
};

// Helper to create a collective config
CollectiveConfig create_test_collective_config(const std::string& name) {
  CollectiveConfig config;
  config.name = name;
  config.inventory_config.limit_defs.push_back(LimitDef({0}, 1000));
  config.inventory_config.limit_defs.push_back(LimitDef({1}, 1000));
  config.inventory_config.limit_defs.push_back(LimitDef({2}, 1000));
  return config;
}

// ============================================================================
// Filter Tests
// ============================================================================

void test_vibe_filter_matches() {
  std::cout << "Testing VibeFilter matches..." << std::endl;

  TestActivationObject actor("actor", 1);    // vibe = 1
  TestActivationObject target("target", 2);  // vibe = 2

  ActivationContext ctx(&actor, &target);

  // Filter for target with vibe_id = 2
  VibeFilterConfig config;
  config.entity = EntityRef::target;
  config.vibe_id = 2;

  VibeFilter filter(config);
  assert(filter.passes(ctx) == true);

  std::cout << "✓ VibeFilter matches test passed" << std::endl;
}

void test_vibe_filter_no_match() {
  std::cout << "Testing VibeFilter no match..." << std::endl;

  TestActivationObject actor("actor", 1);
  TestActivationObject target("target", 3);  // vibe = 3

  ActivationContext ctx(&actor, &target);

  // Filter for target with vibe_id = 2 (doesn't match)
  VibeFilterConfig config;
  config.entity = EntityRef::target;
  config.vibe_id = 2;

  VibeFilter filter(config);
  assert(filter.passes(ctx) == false);

  std::cout << "✓ VibeFilter no match test passed" << std::endl;
}

void test_vibe_filter_actor() {
  std::cout << "Testing VibeFilter on actor..." << std::endl;

  TestActivationObject actor("actor", 5);  // vibe = 5
  TestActivationObject target("target", 0);

  ActivationContext ctx(&actor, &target);

  // Filter for actor with vibe_id = 5
  VibeFilterConfig config;
  config.entity = EntityRef::actor;
  config.vibe_id = 5;

  VibeFilter filter(config);
  assert(filter.passes(ctx) == true);

  std::cout << "✓ VibeFilter on actor test passed" << std::endl;
}

void test_resource_filter_passes() {
  std::cout << "Testing ResourceFilter passes..." << std::endl;

  TestActivationObject actor("actor");
  TestActivationObject target("target");
  target.inventory.update(0, 100);  // 100 health

  ActivationContext ctx(&actor, &target);

  ResourceFilterConfig config;
  config.entity = EntityRef::target;
  config.resource_id = 0;
  config.min_amount = 50;

  ResourceFilter filter(config);
  assert(filter.passes(ctx) == true);

  std::cout << "✓ ResourceFilter passes test passed" << std::endl;
}

void test_resource_filter_fails() {
  std::cout << "Testing ResourceFilter fails..." << std::endl;

  TestActivationObject actor("actor");
  TestActivationObject target("target");
  target.inventory.update(0, 25);  // Only 25 health

  ActivationContext ctx(&actor, &target);

  ResourceFilterConfig config;
  config.entity = EntityRef::target;
  config.resource_id = 0;
  config.min_amount = 50;  // Requires 50

  ResourceFilter filter(config);
  assert(filter.passes(ctx) == false);

  std::cout << "✓ ResourceFilter fails test passed" << std::endl;
}

void test_alignment_filter_same_collective() {
  std::cout << "Testing AlignmentFilter same_collective..." << std::endl;

  CollectiveConfig coll_config = create_test_collective_config("team_a");
  Collective collective(coll_config, &test_resource_names);

  TestActivationObject actor("actor");
  TestActivationObject target("target");

  actor.setCollective(&collective);
  target.setCollective(&collective);

  ActivationContext ctx(&actor, &target);

  AlignmentFilterConfig config;
  config.condition = AlignmentCondition::same_collective;

  AlignmentFilter filter(config);
  assert(filter.passes(ctx) == true);

  std::cout << "✓ AlignmentFilter same_collective test passed" << std::endl;
}

void test_alignment_filter_different_collective() {
  std::cout << "Testing AlignmentFilter different_collective..." << std::endl;

  CollectiveConfig coll_config_a = create_test_collective_config("team_a");
  CollectiveConfig coll_config_b = create_test_collective_config("team_b");
  Collective collective_a(coll_config_a, &test_resource_names);
  Collective collective_b(coll_config_b, &test_resource_names);

  TestActivationObject actor("actor");
  TestActivationObject target("target");

  actor.setCollective(&collective_a);
  target.setCollective(&collective_b);

  ActivationContext ctx(&actor, &target);

  AlignmentFilterConfig config;
  config.condition = AlignmentCondition::different_collective;

  AlignmentFilter filter(config);
  assert(filter.passes(ctx) == true);

  std::cout << "✓ AlignmentFilter different_collective test passed" << std::endl;
}

void test_alignment_filter_unaligned() {
  std::cout << "Testing AlignmentFilter unaligned..." << std::endl;

  TestActivationObject actor("actor");
  TestActivationObject target("target");
  // Neither has a collective

  ActivationContext ctx(&actor, &target);

  AlignmentFilterConfig config;
  config.condition = AlignmentCondition::unaligned;

  AlignmentFilter filter(config);
  assert(filter.passes(ctx) == true);

  std::cout << "✓ AlignmentFilter unaligned test passed" << std::endl;
}

void test_tag_filter_matches() {
  std::cout << "Testing TagFilter matches..." << std::endl;

  TestActivationObject actor("actor");
  TestActivationObject target("target");
  target.tag_ids.push_back(42);
  target.tag_ids.push_back(100);

  ActivationContext ctx(&actor, &target);

  TagFilterConfig config;
  config.entity = EntityRef::target;
  config.required_tag_ids.push_back(42);

  TagFilter filter(config);
  assert(filter.passes(ctx) == true);

  std::cout << "✓ TagFilter matches test passed" << std::endl;
}

void test_tag_filter_no_match() {
  std::cout << "Testing TagFilter no match..." << std::endl;

  TestActivationObject actor("actor");
  TestActivationObject target("target");
  target.tag_ids.push_back(1);
  target.tag_ids.push_back(2);

  ActivationContext ctx(&actor, &target);

  TagFilterConfig config;
  config.entity = EntityRef::target;
  config.required_tag_ids.push_back(42);  // Target doesn't have tag 42

  TagFilter filter(config);
  assert(filter.passes(ctx) == false);

  std::cout << "✓ TagFilter no match test passed" << std::endl;
}

void test_tag_filter_empty_required() {
  std::cout << "Testing TagFilter empty required tags..." << std::endl;

  TestActivationObject actor("actor");
  TestActivationObject target("target");

  ActivationContext ctx(&actor, &target);

  TagFilterConfig config;
  config.entity = EntityRef::target;
  // No required tags - should pass

  TagFilter filter(config);
  assert(filter.passes(ctx) == true);

  std::cout << "✓ TagFilter empty required tags test passed" << std::endl;
}

// ============================================================================
// Mutation Tests
// ============================================================================

void test_resource_delta_mutation_add() {
  std::cout << "Testing ResourceDeltaMutation add..." << std::endl;

  TestActivationObject actor("actor");
  TestActivationObject target("target");
  target.inventory.update(0, 100);  // Start with 100 health

  ActivationContext ctx(&actor, &target);

  ResourceDeltaMutationConfig config;
  config.entity = EntityRef::target;
  config.resource_id = 0;
  config.delta = 50;

  ResourceDeltaMutation mutation(config);
  mutation.apply(ctx);

  assert(target.inventory.amount(0) == 150);

  std::cout << "✓ ResourceDeltaMutation add test passed" << std::endl;
}

void test_resource_delta_mutation_subtract() {
  std::cout << "Testing ResourceDeltaMutation subtract..." << std::endl;

  TestActivationObject actor("actor");
  TestActivationObject target("target");
  target.inventory.update(0, 100);

  ActivationContext ctx(&actor, &target);

  ResourceDeltaMutationConfig config;
  config.entity = EntityRef::target;
  config.resource_id = 0;
  config.delta = -30;

  ResourceDeltaMutation mutation(config);
  mutation.apply(ctx);

  assert(target.inventory.amount(0) == 70);

  std::cout << "✓ ResourceDeltaMutation subtract test passed" << std::endl;
}

void test_resource_transfer_mutation() {
  std::cout << "Testing ResourceTransferMutation..." << std::endl;

  TestActivationObject actor("actor");
  TestActivationObject target("target");
  actor.inventory.update(1, 100);  // Actor has 100 energy

  ActivationContext ctx(&actor, &target);

  ResourceTransferMutationConfig config;
  config.source = EntityRef::actor;
  config.destination = EntityRef::target;
  config.resource_id = 1;
  config.amount = 40;

  ResourceTransferMutation mutation(config);
  mutation.apply(ctx);

  assert(actor.inventory.amount(1) == 60);   // 100 - 40
  assert(target.inventory.amount(1) == 40);  // 0 + 40

  std::cout << "✓ ResourceTransferMutation test passed" << std::endl;
}

void test_resource_transfer_mutation_all() {
  std::cout << "Testing ResourceTransferMutation transfer all..." << std::endl;

  TestActivationObject actor("actor");
  TestActivationObject target("target");
  actor.inventory.update(2, 75);  // Actor has 75 gold

  ActivationContext ctx(&actor, &target);

  ResourceTransferMutationConfig config;
  config.source = EntityRef::actor;
  config.destination = EntityRef::target;
  config.resource_id = 2;
  config.amount = -1;  // Transfer all

  ResourceTransferMutation mutation(config);
  mutation.apply(ctx);

  assert(actor.inventory.amount(2) == 0);    // All transferred
  assert(target.inventory.amount(2) == 75);  // Received all

  std::cout << "✓ ResourceTransferMutation transfer all test passed" << std::endl;
}

void test_alignment_mutation_to_actor_collective() {
  std::cout << "Testing AlignmentMutation to actor collective..." << std::endl;

  CollectiveConfig coll_config = create_test_collective_config("team_a");
  Collective collective(coll_config, &test_resource_names);

  TestActivationObject actor("actor");
  TestActivationObject target("target");

  actor.setCollective(&collective);
  // Target has no collective initially

  ActivationContext ctx(&actor, &target);

  AlignmentMutationConfig config;
  config.align_to = AlignTo::actor_collective;

  AlignmentMutation mutation(config);
  mutation.apply(ctx);

  assert(target.getCollective() == &collective);

  std::cout << "✓ AlignmentMutation to actor collective test passed" << std::endl;
}

void test_alignment_mutation_to_none() {
  std::cout << "Testing AlignmentMutation to none..." << std::endl;

  CollectiveConfig coll_config = create_test_collective_config("team_a");
  Collective collective(coll_config, &test_resource_names);

  TestActivationObject actor("actor");
  TestActivationObject target("target");

  target.setCollective(&collective);

  ActivationContext ctx(&actor, &target);

  AlignmentMutationConfig config;
  config.align_to = AlignTo::none;

  AlignmentMutation mutation(config);
  mutation.apply(ctx);

  assert(target.getCollective() == nullptr);

  std::cout << "✓ AlignmentMutation to none test passed" << std::endl;
}

void test_clear_inventory_mutation_specific() {
  std::cout << "Testing ClearInventoryMutation specific resource..." << std::endl;

  TestActivationObject actor("actor");
  TestActivationObject target("target");
  target.inventory.update(0, 100);  // health
  target.inventory.update(1, 50);   // energy
  target.inventory.update(2, 25);   // gold

  ActivationContext ctx(&actor, &target);

  ClearInventoryMutationConfig config;
  config.entity = EntityRef::target;
  config.resource_id = 1;  // Clear only energy

  ClearInventoryMutation mutation(config);
  mutation.apply(ctx);

  assert(target.inventory.amount(0) == 100);  // Unchanged
  assert(target.inventory.amount(1) == 0);    // Cleared
  assert(target.inventory.amount(2) == 25);   // Unchanged

  std::cout << "✓ ClearInventoryMutation specific resource test passed" << std::endl;
}

void test_clear_inventory_mutation_all() {
  std::cout << "Testing ClearInventoryMutation all resources..." << std::endl;

  TestActivationObject actor("actor");
  TestActivationObject target("target");
  target.inventory.update(0, 100);
  target.inventory.update(1, 50);
  target.inventory.update(2, 25);

  ActivationContext ctx(&actor, &target);

  ClearInventoryMutationConfig config;
  config.entity = EntityRef::target;
  config.resource_id = 255;  // Clear all

  ClearInventoryMutation mutation(config);
  mutation.apply(ctx);

  assert(target.inventory.amount(0) == 0);
  assert(target.inventory.amount(1) == 0);
  assert(target.inventory.amount(2) == 0);

  std::cout << "✓ ClearInventoryMutation all resources test passed" << std::endl;
}

void test_attack_mutation() {
  std::cout << "Testing AttackMutation..." << std::endl;

  TestActivationObject actor("actor");
  TestActivationObject target("target");

  actor.inventory.update(0, 10);   // Weapon power = 10
  target.inventory.update(1, 3);   // Armor = 3
  target.inventory.update(2, 50);  // Health = 50

  ActivationContext ctx(&actor, &target);

  AttackMutationConfig config;
  config.weapon_resource = 0;
  config.armor_resource = 1;
  config.health_resource = 2;
  config.damage_multiplier = 1.0f;

  AttackMutation mutation(config);
  mutation.apply(ctx);

  // Damage = (10 * 1.0) - 3 = 7
  // Health = 50 - 7 = 43
  assert(target.inventory.amount(2) == 43);

  std::cout << "✓ AttackMutation test passed" << std::endl;
}

// ============================================================================
// ActivationHandler Tests
// ============================================================================

void test_activation_handler_filters_pass() {
  std::cout << "Testing ActivationHandler filters pass..." << std::endl;

  CollectiveConfig coll_config = create_test_collective_config("team_a");
  Collective collective(coll_config, &test_resource_names);

  TestActivationObject actor("actor");
  TestActivationObject target("target");
  actor.setCollective(&collective);
  target.setCollective(&collective);
  target.inventory.update(0, 100);

  // Create handler config with alignment and resource filters
  ActivationHandlerConfig handler_config("test_handler");

  AlignmentFilterConfig align_filter;
  align_filter.condition = AlignmentCondition::same_collective;
  handler_config.filters.push_back(align_filter);

  ResourceFilterConfig resource_filter;
  resource_filter.entity = EntityRef::target;
  resource_filter.resource_id = 0;
  resource_filter.min_amount = 50;
  handler_config.filters.push_back(resource_filter);

  // Add a mutation
  ResourceDeltaMutationConfig delta_mutation;
  delta_mutation.entity = EntityRef::target;
  delta_mutation.resource_id = 0;
  delta_mutation.delta = -25;
  handler_config.mutations.push_back(delta_mutation);

  ActivationHandler handler(handler_config);
  bool result = handler.try_apply(&actor, &target);

  assert(result == true);
  assert(target.inventory.amount(0) == 75);  // 100 - 25

  std::cout << "✓ ActivationHandler filters pass test passed" << std::endl;
}

void test_activation_handler_filters_fail() {
  std::cout << "Testing ActivationHandler filters fail..." << std::endl;

  CollectiveConfig coll_config_a = create_test_collective_config("team_a");
  CollectiveConfig coll_config_b = create_test_collective_config("team_b");
  Collective collective_a(coll_config_a, &test_resource_names);
  Collective collective_b(coll_config_b, &test_resource_names);

  TestActivationObject actor("actor");
  TestActivationObject target("target");
  actor.setCollective(&collective_a);
  target.setCollective(&collective_b);  // Different collective
  target.inventory.update(0, 100);

  // Create handler config with same_collective filter
  ActivationHandlerConfig handler_config("test_handler");

  AlignmentFilterConfig align_filter;
  align_filter.condition = AlignmentCondition::same_collective;  // Will fail
  handler_config.filters.push_back(align_filter);

  ResourceDeltaMutationConfig delta_mutation;
  delta_mutation.entity = EntityRef::target;
  delta_mutation.resource_id = 0;
  delta_mutation.delta = -25;
  handler_config.mutations.push_back(delta_mutation);

  ActivationHandler handler(handler_config);
  bool result = handler.try_apply(&actor, &target);

  assert(result == false);
  assert(target.inventory.amount(0) == 100);  // Unchanged

  std::cout << "✓ ActivationHandler filters fail test passed" << std::endl;
}

void test_activation_handler_multiple_mutations() {
  std::cout << "Testing ActivationHandler multiple mutations..." << std::endl;

  TestActivationObject actor("actor");
  TestActivationObject target("target");
  actor.inventory.update(2, 100);  // Actor has gold
  target.inventory.update(0, 50);  // Target has health

  ActivationHandlerConfig handler_config("multi_mutation_handler");

  // Mutation 1: Transfer gold from actor to target
  ResourceTransferMutationConfig transfer;
  transfer.source = EntityRef::actor;
  transfer.destination = EntityRef::target;
  transfer.resource_id = 2;
  transfer.amount = 30;
  handler_config.mutations.push_back(transfer);

  // Mutation 2: Add health to target
  ResourceDeltaMutationConfig heal;
  heal.entity = EntityRef::target;
  heal.resource_id = 0;
  heal.delta = 20;
  handler_config.mutations.push_back(heal);

  ActivationHandler handler(handler_config);
  bool result = handler.try_apply(&actor, &target);

  assert(result == true);
  assert(actor.inventory.amount(2) == 70);   // 100 - 30
  assert(target.inventory.amount(2) == 30);  // 0 + 30
  assert(target.inventory.amount(0) == 70);  // 50 + 20

  std::cout << "✓ ActivationHandler multiple mutations test passed" << std::endl;
}

void test_activation_handler_check_filters_only() {
  std::cout << "Testing ActivationHandler check_filters..." << std::endl;

  TestActivationObject actor("actor");
  TestActivationObject target("target");
  target.inventory.update(0, 100);

  ActivationHandlerConfig handler_config("test_handler");

  ResourceFilterConfig resource_filter;
  resource_filter.entity = EntityRef::target;
  resource_filter.resource_id = 0;
  resource_filter.min_amount = 50;
  handler_config.filters.push_back(resource_filter);

  ResourceDeltaMutationConfig delta_mutation;
  delta_mutation.entity = EntityRef::target;
  delta_mutation.resource_id = 0;
  delta_mutation.delta = -25;
  handler_config.mutations.push_back(delta_mutation);

  ActivationHandler handler(handler_config);

  // check_filters should pass but NOT apply mutations
  bool can_apply = handler.check_filters(&actor, &target);
  assert(can_apply == true);
  assert(target.inventory.amount(0) == 100);  // Still unchanged

  std::cout << "✓ ActivationHandler check_filters test passed" << std::endl;
}

int main() {
  std::cout << "Running ActivationHandler tests..." << std::endl;
  std::cout << "================================================" << std::endl;

  // Filter tests
  test_vibe_filter_matches();
  test_vibe_filter_no_match();
  test_vibe_filter_actor();
  test_resource_filter_passes();
  test_resource_filter_fails();
  test_alignment_filter_same_collective();
  test_alignment_filter_different_collective();
  test_alignment_filter_unaligned();
  test_tag_filter_matches();
  test_tag_filter_no_match();
  test_tag_filter_empty_required();

  // Mutation tests
  test_resource_delta_mutation_add();
  test_resource_delta_mutation_subtract();
  test_resource_transfer_mutation();
  test_resource_transfer_mutation_all();
  test_alignment_mutation_to_actor_collective();
  test_alignment_mutation_to_none();
  test_clear_inventory_mutation_specific();
  test_clear_inventory_mutation_all();
  test_attack_mutation();

  // Handler tests
  test_activation_handler_filters_pass();
  test_activation_handler_filters_fail();
  test_activation_handler_multiple_mutations();
  test_activation_handler_check_filters_only();

  std::cout << "================================================" << std::endl;
  std::cout << "All ActivationHandler tests passed! ✓" << std::endl;

  return 0;
}
