#include "observation_encoder.hpp"

#include <gtest/gtest.h>

#include <memory>
#include <string>
#include <vector>

#include "grid_object.hpp"

// Mock GridObject classes for testing different object types
class WallObject : public GridObject {
public:
  WallObject(unsigned int id) {
    this->id = id;
    this->_type_id = 1;  // Type 1 for walls
    this->location = GridLocation(0, 0, 0);
  }

  void obs(ObsType* observation, const std::vector<unsigned int>& offsets) const override {
    observation[offsets[0]] = 1;    // Type ID
    observation[offsets[1]] = 255;  // Full opacity
  }
};

class AgentObject : public GridObject {
public:
  AgentObject(unsigned int id, unsigned int team_id) {
    this->id = id;
    this->_type_id = 2;  // Type 2 for agents
    this->location = GridLocation(0, 0, 0);
    this->team_id = team_id;
  }

  void obs(ObsType* observation, const std::vector<unsigned int>& offsets) const override {
    observation[offsets[0]] = 2;        // Type ID
    observation[offsets[1]] = 150;      // Medium opacity
    observation[offsets[2]] = team_id;  // Team ID
  }

  unsigned int team_id;
};

class ResourceObject : public GridObject {
public:
  ResourceObject(unsigned int id, unsigned int resource_amount) {
    this->id = id;
    this->_type_id = 3;  // Type 3 for resources
    this->location = GridLocation(0, 0, 0);
    this->resource_amount = resource_amount;
  }

  void obs(ObsType* observation, const std::vector<unsigned int>& offsets) const override {
    observation[offsets[0]] = 3;                // Type ID
    observation[offsets[1]] = 100;              // Lower opacity
    observation[offsets[2]] = resource_amount;  // Resource amount
  }

  unsigned int resource_amount;
};

// Custom ObservationEncoder for testing
class TestObservationEncoder : public ObservationEncoder {
public:
  TestObservationEncoder() {}

  std::vector<std::string> feature_names() const override {
    return {"type", "opacity", "attribute1", "attribute2"};
  }

  void encode(const GridObject* obj, ObsType* observation) override {
    std::vector<unsigned int> offsets = {0, 1, 2, 3};
    obj->obs(observation, offsets);
  }
};

// Test fixture for ObservationEncoder
class ObservationEncoderTest : public ::testing::Test {
protected:
  std::unique_ptr<TestObservationEncoder> encoder;
  WallObject* wall;
  AgentObject* agent;
  ResourceObject* resource;
  ObsType* observation;

  void SetUp() override {
    encoder = std::make_unique<TestObservationEncoder>();
    wall = new WallObject(1);
    agent = new AgentObject(2, 1);
    resource = new ResourceObject(3, 42);

    // Allocate observation buffer (4 features)
    observation = new ObsType[4];
  }

  void TearDown() override {
    delete[] observation;
    delete wall;
    delete agent;
    delete resource;
  }
};

// Test feature names
TEST_F(ObservationEncoderTest, FeatureNames) {
  auto features = encoder->feature_names();
  EXPECT_EQ(4, features.size());
  EXPECT_EQ("type", features[0]);
  EXPECT_EQ("opacity", features[1]);
  EXPECT_EQ("attribute1", features[2]);
  EXPECT_EQ("attribute2", features[3]);
}

// Test encoding of wall object
TEST_F(ObservationEncoderTest, EncodeWall) {
  // Clear observation buffer
  memset(observation, 0, 4 * sizeof(ObsType));

  // Encode wall object
  encoder->encode(wall, observation);

  // Check encoded values
  EXPECT_EQ(1, observation[0]);    // Type ID
  EXPECT_EQ(255, observation[1]);  // Opacity
  EXPECT_EQ(0, observation[2]);    // Unused attribute1
  EXPECT_EQ(0, observation[3]);    // Unused attribute2
}

// Test encoding of agent object
TEST_F(ObservationEncoderTest, EncodeAgent) {
  // Clear observation buffer
  memset(observation, 0, 4 * sizeof(ObsType));

  // Encode agent object
  encoder->encode(agent, observation);

  // Check encoded values
  EXPECT_EQ(2, observation[0]);    // Type ID
  EXPECT_EQ(150, observation[1]);  // Opacity
  EXPECT_EQ(1, observation[2]);    // Team ID
  EXPECT_EQ(0, observation[3]);    // Unused attribute2
}

// Test encoding of resource object
TEST_F(ObservationEncoderTest, EncodeResource) {
  // Clear observation buffer
  memset(observation, 0, 4 * sizeof(ObsType));

  // Encode resource object
  encoder->encode(resource, observation);

  // Check encoded values
  EXPECT_EQ(3, observation[0]);    // Type ID
  EXPECT_EQ(100, observation[1]);  // Opacity
  EXPECT_EQ(42, observation[2]);   // Resource amount
  EXPECT_EQ(0, observation[3]);    // Unused attribute2
}

// Test encoding multiple objects
TEST_F(ObservationEncoderTest, EncodeMultipleObjects) {
  // Create a batch of objects
  std::vector<GridObject*> objects = {wall, agent, resource};

  // Encode each object into its own observation buffer
  std::vector<ObsType*> observations;
  for (size_t i = 0; i < objects.size(); i++) {
    ObsType* obj_obs = new ObsType[4];
    memset(obj_obs, 0, 4 * sizeof(ObsType));
    encoder->encode(objects[i], obj_obs);
    observations.push_back(obj_obs);
  }

  // Check wall observation
  EXPECT_EQ(1, observations[0][0]);    // Type ID
  EXPECT_EQ(255, observations[0][1]);  // Opacity

  // Check agent observation
  EXPECT_EQ(2, observations[1][0]);    // Type ID
  EXPECT_EQ(150, observations[1][1]);  // Opacity
  EXPECT_EQ(1, observations[1][2]);    // Team ID

  // Check resource observation
  EXPECT_EQ(3, observations[2][0]);    // Type ID
  EXPECT_EQ(100, observations[2][1]);  // Opacity
  EXPECT_EQ(42, observations[2][2]);   // Resource amount

  // Clean up
  for (auto obs : observations) {
    delete[] obs;
  }
}

// Test extended observation encoder with more features
class ExtendedObservationEncoder : public ObservationEncoder {
public:
  ExtendedObservationEncoder() {}

  std::vector<std::string> feature_names() const override {
    return {"type", "opacity", "attribute1", "attribute2", "attribute3", "attribute4"};
  }

  void encode(const GridObject* obj, ObsType* observation) override {
    // Basic encoding of common attributes
    std::vector<unsigned int> offsets = {0, 1, 2, 3};
    obj->obs(observation, offsets);

    // Extended attributes based on object type
    if (obj->_type_id == 1) {         // Wall
      observation[4] = 1;             // Blocking
      observation[5] = 0;             // No interaction
    } else if (obj->_type_id == 2) {  // Agent
      observation[4] = 0;             // Non-blocking
      observation[5] = 1;             // Can interact
    } else if (obj->_type_id == 3) {  // Resource
      observation[4] = 0;             // Non-blocking
      observation[5] = 2;             // Collectible
    }
  }
};

// Test fixture for ExtendedObservationEncoder
class ExtendedObservationEncoderTest : public ::testing::Test {
protected:
  std::unique_ptr<ExtendedObservationEncoder> ext_encoder;
  WallObject* wall;
  AgentObject* agent;
  ResourceObject* resource;
  ObsType* observation;

  void SetUp() override {
    ext_encoder = std::make_unique<ExtendedObservationEncoder>();
    wall = new WallObject(1);
    agent = new AgentObject(2, 1);
    resource = new ResourceObject(3, 42);

    // Allocate observation buffer (6 features)
    observation = new ObsType[6];
  }

  void TearDown() override {
    delete[] observation;
    delete wall;
    delete agent;
    delete resource;
  }
};

// Test extended feature names
TEST_F(ExtendedObservationEncoderTest, ExtendedFeatureNames) {
  auto features = ext_encoder->feature_names();
  EXPECT_EQ(6, features.size());
  EXPECT_EQ("type", features[0]);
  EXPECT_EQ("opacity", features[1]);
  EXPECT_EQ("attribute1", features[2]);
  EXPECT_EQ("attribute2", features[3]);
  EXPECT_EQ("attribute3", features[4]);
  EXPECT_EQ("attribute4", features[5]);
}

// Test extended encoding of wall object
TEST_F(ExtendedObservationEncoderTest, ExtendedEncodeWall) {
  // Clear observation buffer
  memset(observation, 0, 6 * sizeof(ObsType));

  // Encode wall object
  ext_encoder->encode(wall, observation);

  // Check encoded values
  EXPECT_EQ(1, observation[0]);    // Type ID
  EXPECT_EQ(255, observation[1]);  // Opacity
  EXPECT_EQ(0, observation[2]);    // Unused attribute1
  EXPECT_EQ(0, observation[3]);    // Unused attribute2
  EXPECT_EQ(1, observation[4]);    // Blocking
  EXPECT_EQ(0, observation[5]);    // No interaction
}

// Test extended encoding of agent object
TEST_F(ExtendedObservationEncoderTest, ExtendedEncodeAgent) {
  // Clear observation buffer
  memset(observation, 0, 6 * sizeof(ObsType));

  // Encode agent object
  ext_encoder->encode(agent, observation);

  // Check encoded values
  EXPECT_EQ(2, observation[0]);    // Type ID
  EXPECT_EQ(150, observation[1]);  // Opacity
  EXPECT_EQ(1, observation[2]);    // Team ID
  EXPECT_EQ(0, observation[3]);    // Unused attribute2
  EXPECT_EQ(0, observation[4]);    // Non-blocking
  EXPECT_EQ(1, observation[5]);    // Can interact
}

// Test extended encoding of resource object
TEST_F(ExtendedObservationEncoderTest, ExtendedEncodeResource) {
  // Clear observation buffer
  memset(observation, 0, 6 * sizeof(ObsType));

  // Encode resource object
  ext_encoder->encode(resource, observation);

  // Check encoded values
  EXPECT_EQ(3, observation[0]);    // Type ID
  EXPECT_EQ(100, observation[1]);  // Opacity
  EXPECT_EQ(42, observation[2]);   // Resource amount
  EXPECT_EQ(0, observation[3]);    // Unused attribute2
  EXPECT_EQ(0, observation[4]);    // Non-blocking
  EXPECT_EQ(2, observation[5]);    // Collectible
}

// Run the tests
int main(int argc, char** argv) {
  ::testing::InitGoogleTest(&argc, argv);
  return RUN_ALL_TEST();
}