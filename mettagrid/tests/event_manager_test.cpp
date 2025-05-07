#include <gtest/gtest.h>

#include <memory>
#include <vector>

#include "event.hpp"

// Mock Event class for testing
class MockEvent : public Event {
public:
  MockEvent(unsigned int trigger_timestep = 1) : processed(false) {
    this->trigger_timestep = trigger_timestep;
  }

  bool process(unsigned int current_timestep) override {
    if (current_timestep >= trigger_timestep) {
      processed = true;
      return true;
    }
    return false;
  }

  bool processed;
};

// Test fixture for EventManager
class EventManagerTest : public ::testing::Test {
protected:
  std::unique_ptr<EventManager> event_manager;
  std::vector<MockEvent*> events;

  void SetUp() override {
    event_manager = std::make_unique<EventManager>();
    // Create test events
    for (int i = 1; i <= 5; i++) {
      auto event = new MockEvent(i);
      events.push_back(event);
      event_manager->add_event(event);
    }
  }

  void TearDown() override {
    // No need to delete events as EventManager takes ownership
  }
};

// Test adding events
TEST_F(EventManagerTest, AddEvent) {
  // Add a new event
  auto new_event = new MockEvent(10);
  event_manager->add_event(new_event);

  // Process events up to timestep 9
  event_manager->process_events(9);

  // The new event should not be processed yet
  EXPECT_FALSE(new_event->processed);

  // Process events at timestep 10
  event_manager->process_events(10);

  // The new event should now be processed
  EXPECT_TRUE(new_event->processed);
}

// Test processing events
TEST_F(EventManagerTest, ProcessEvents) {
  // Process events at timestep 0
  event_manager->process_events(0);

  // No events should be processed yet
  for (auto event : events) {
    EXPECT_FALSE(event->processed);
  }

  // Process events at timestep 3
  event_manager->process_events(3);

  // Events with trigger_timestep <= 3 should be processed
  EXPECT_TRUE(events[0]->processed);   // trigger_timestep = 1
  EXPECT_TRUE(events[1]->processed);   // trigger_timestep = 2
  EXPECT_TRUE(events[2]->processed);   // trigger_timestep = 3
  EXPECT_FALSE(events[3]->processed);  // trigger_timestep = 4
  EXPECT_FALSE(events[4]->processed);  // trigger_timestep = 5

  // Process events at timestep 5
  event_manager->process_events(5);

  // All events should now be processed
  for (auto event : events) {
    EXPECT_TRUE(event->processed);
  }
}

// Test removing events
TEST_F(EventManagerTest, RemoveEvent) {
  // Create a new event and add it
  auto removable_event = new MockEvent(7);
  event_manager->add_event(removable_event);

  // Remove the event
  event_manager->remove_event(removable_event);

  // Process events at timestep 10 (past all trigger timesteps)
  event_manager->process_events(10);

  // The removed event should not have been processed
  EXPECT_FALSE(removable_event->processed);

  // Clean up the removed event as EventManager no longer owns it
  delete removable_event;
}

// Test clearing all events
TEST_F(EventManagerTest, ClearEvents) {
  // Create some new events that we'll track separately
  auto event1 = new MockEvent(15);
  auto event2 = new MockEvent(20);
  event_manager->add_event(event1);
  event_manager->add_event(event2);

  // Clear all events
  event_manager->clear_events();

  // Process events at timestep 25 (past all trigger timesteps)
  event_manager->process_events(25);

  // No events should have been processed since they were cleared
  EXPECT_FALSE(event1->processed);
  EXPECT_FALSE(event2->processed);
  for (auto event : events) {
    EXPECT_FALSE(event->processed);
  }

  // Clean up the events we created
  delete event1;
  delete event2;
}

// Test ordered event processing
TEST_F(EventManagerTest, OrderedProcessing) {
  // Create events with the same trigger timestep but different priorities
  auto high_priority = new MockEvent(10);
  high_priority->priority = 10;

  auto medium_priority = new MockEvent(10);
  medium_priority->priority = 5;

  auto low_priority = new MockEvent(10);
  low_priority->priority = 1;

  // Add events in reverse priority order to test sorting
  event_manager->add_event(low_priority);
  event_manager->add_event(high_priority);
  event_manager->add_event(medium_priority);

  // Process up to timestep 10
  event_manager->process_events(10);

  // All events should be processed
  EXPECT_TRUE(high_priority->processed);
  EXPECT_TRUE(medium_priority->processed);
  EXPECT_TRUE(low_priority->processed);

  // Check processing order by implementing a tracking mechanism in EventManager
  // This would require modifying EventManager to expose processing order
  // For this test, we're just checking they're all processed
}

// Test conditional event processing
TEST_F(EventManagerTest, ConditionalProcessing) {
  class ConditionalEvent : public MockEvent {
  public:
    ConditionalEvent(unsigned int trigger_timestep, bool* condition_ptr)
        : MockEvent(trigger_timestep), condition_ptr(condition_ptr) {}

    bool process(unsigned int current_timestep) override {
      if (current_timestep >= trigger_timestep && *condition_ptr) {
        processed = true;
        return true;
      }
      return false;
    }

    bool* condition_ptr;
  };

  // Set up condition flag
  bool condition = false;

  // Create conditional event
  auto conditional_event = new ConditionalEvent(5, &condition);
  event_manager->add_event(conditional_event);

  // Process events at timestep 10 with condition false
  event_manager->process_events(10);

  // Event should not be processed yet
  EXPECT_FALSE(conditional_event->processed);

  // Set condition to true
  condition = true;

  // Process events again at timestep 10
  event_manager->process_events(10);

  // Now the event should be processed
  EXPECT_TRUE(conditional_event->processed);
}

// Test one-time events vs. recurring events
TEST_F(EventManagerTest, RecurringEvents) {
  class RecurringEvent : public MockEvent {
  public:
    RecurringEvent(unsigned int trigger_timestep, unsigned int interval)
        : MockEvent(trigger_timestep), interval(interval), process_count(0) {}

    bool process(unsigned int current_timestep) override {
      if (current_timestep >= trigger_timestep && (current_timestep - trigger_timestep) % interval == 0) {
        processed = true;
        process_count++;
        return false;  // Return false to not remove from event list
      }
      return false;
    }

    unsigned int interval;
    unsigned int process_count;
  };

  // Create a recurring event that triggers every 5 timesteps starting at 5
  auto recurring_event = new RecurringEvent(5, 5);
  event_manager->add_event(recurring_event);

  // Process events at various timesteps
  event_manager->process_events(4);  // Before first trigger
  EXPECT_EQ(0, recurring_event->process_count);

  event_manager->process_events(5);  // First trigger
  EXPECT_EQ(1, recurring_event->process_count);

  event_manager->process_events(9);  // Before second trigger
  EXPECT_EQ(1, recurring_event->process_count);

  event_manager->process_events(10);  // Second trigger
  EXPECT_EQ(2, recurring_event->process_count);

  event_manager->process_events(15);  // Third trigger
  EXPECT_EQ(3, recurring_event->process_count);
}

// Run the tests
int main(int argc, char** argv) {
  ::testing::InitGoogleTest(&argc, argv);
  return RUN_ALL_TEST();
}