.PHONY: clean build test benchmark

.DEFAULT_GOAL := build

# Note: you don't need to run `make` if you just want to build mettagrid.  You
# can run `pip install -e .` instead. It will build MettaGrid in an isolated
# environment, through scikit-build-core.
# 
# But if you need to run C++ tests or benchmarks, you'll need to `make` and
# then run `make test` or `make benchmark`.  It will build tests to `./tests/`
# and benchmarks to `./benchmarks/`.
build:
	@cmake -B build -DBUILD_TESTS=ON
	@cmake --build build

clean:
	@echo "(MettaGrid) Cleaning build artifacts..."
	@[ -d build ] && cmake --build build --target clean || true
	@rm -rf build
	@echo "(MettaGrid) Clean completed successfully"

test:
	@ctest --test-dir build

benchmark:
	@./benchmarks/grid_object_benchmark
