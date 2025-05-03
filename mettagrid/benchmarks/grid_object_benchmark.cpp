#include <benchmark/benchmark.h>

#include <random>
#include <vector>

#include "grid_object.hpp"

// Define a concrete implementation of GridObject for benchmarking
class TestGridObject : public GridObject {
public:
  TestGridObject() = default;

  TestGridObject(TypeId type_id, GridCoord r, GridCoord c, Layer layer) {
    init(type_id, r, c, layer);
  }

  void obs(ObsType* obs, const std::vector<unsigned int>& offsets) const override {
    // Simple implementation for benchmarking
    for (size_t i = 0; i < offsets.size(); ++i) {
      obs[offsets[i]] = i % 255;
    }
  }
};

// Benchmark initializing a GridLocation object
static void BM_GridLocationCreation(benchmark::State& state) {
  for (auto _ : state) {
    GridLocation loc(10, 20, 1);
    benchmark::DoNotOptimize(&loc);
    benchmark::ClobberMemory();
  }
}
BENCHMARK(BM_GridLocationCreation);

// Benchmark initializing a GridObject with GridLocation
static void BM_GridObjectInitWithLocation(benchmark::State& state) {
  for (auto _ : state) {
    TestGridObject obj;
    GridLocation loc(10, 20, 1);
    obj.init(5, loc);
    benchmark::DoNotOptimize(&obj);
    benchmark::ClobberMemory();
  }
}
BENCHMARK(BM_GridObjectInitWithLocation);

// Benchmark initializing a GridObject with coordinates
static void BM_GridObjectInitWithCoordinates(benchmark::State& state) {
  for (auto _ : state) {
    TestGridObject obj;
    obj.init(5, 10, 20);
    benchmark::DoNotOptimize(&obj);
    benchmark::ClobberMemory();
  }
}
BENCHMARK(BM_GridObjectInitWithCoordinates);

// Benchmark initializing a GridObject with coordinates and layer
static void BM_GridObjectInitWithCoordinatesAndLayer(benchmark::State& state) {
  for (auto _ : state) {
    TestGridObject obj;
    obj.init(5, 10, 20, 1);
    benchmark::DoNotOptimize(&obj);
    benchmark::ClobberMemory();
  }
}
BENCHMARK(BM_GridObjectInitWithCoordinatesAndLayer);

// Benchmark the performance of the obs method with varying number of offsets
static void BM_GridObjectObs(benchmark::State& state) {
  // Use the loop iteration count to vary the size of the offsets vector
  const int numOffsets = state.range(0);

  // Set up the test object
  TestGridObject obj(1, 5, 10, 0);

  // Create offsets vector and observation buffer
  std::vector<unsigned int> offsets(numOffsets);
  for (int i = 0; i < numOffsets; ++i) {
    offsets[i] = i;
  }

  std::vector<ObsType> observations(numOffsets, 0);

  for (auto _ : state) {
    obj.obs(observations.data(), offsets);
    benchmark::DoNotOptimize(observations.data());
    benchmark::ClobberMemory();
  }
}
// Test with different sizes of observation vectors
BENCHMARK(BM_GridObjectObs)->Range(1, 1 << 10);  // From 1 to 1024 offsets

// Benchmark creating and initializing many objects
static void BM_CreateManyObjects(benchmark::State& state) {
  const int numObjects = state.range(0);

  // Random number generator for position
  std::random_device rd;
  std::mt19937 gen(rd());
  std::uniform_int_distribution<GridCoord> coordDist(0, 999);
  std::uniform_int_distribution<TypeId> typeDist(0, 10);
  std::uniform_int_distribution<Layer> layerDist(0, 3);

  for (auto _ : state) {
    state.PauseTiming();  // Don't time the vector allocation
    std::vector<TestGridObject> objects(numObjects);
    state.ResumeTiming();

    for (int i = 0; i < numObjects; ++i) {
      objects[i].init(typeDist(gen), coordDist(gen), coordDist(gen), layerDist(gen));
    }

    benchmark::DoNotOptimize(objects.data());
    benchmark::ClobberMemory();
  }

  state.SetItemsProcessed(numObjects * state.iterations());
}
BENCHMARK(BM_CreateManyObjects)->Range(1, 1 << 14);  // From 1 to 16384 objects

// Very simplified benchmark for grid lookups
static void BM_GridLookup(benchmark::State& state) {
  const int size = state.range(0);

  // Create a simple array to simulate grid lookups
  std::vector<int> grid(size * size, 0);
  for (int i = 0; i < size * size; ++i) {
    grid[i] = i;
  }

  // Create random indices to access
  std::random_device rd;
  std::mt19937 gen(rd());
  std::uniform_int_distribution<int> dist(0, size * size - 1);

  for (auto _ : state) {
    int idx = dist(gen);
    int value = grid[idx];
    benchmark::DoNotOptimize(value);
  }

  state.SetItemsProcessed(state.iterations());
}
BENCHMARK(BM_GridLookup)->Range(8, 1 << 10);

BENCHMARK_MAIN();