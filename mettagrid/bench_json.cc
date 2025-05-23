#include <benchmark/benchmark.h>

static void BM_SimpleFunction(benchmark::State& state) {
  for (auto _ : state) {
    // Simulate some work
    int x = 0;
    for (int i = 0; i < 1000; ++i) {
      x += i;
    }
    benchmark::DoNotOptimize(x);
  }
}

BENCHMARK(BM_SimpleFunction);

BENCHMARK_MAIN();
