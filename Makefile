.PHONY: test bench build nvcc-check

# Build release binary
build:
	cargo build --release

# Run all tests
test:
	cargo test

# Run benchmarks
bench:
	cargo bench --bench kernel_params

# Check CUDA kernel stack usage
nvcc-check:
	nvcc --ptxas-options=-v src/hashers/cuda/sha1_optimized.cu 2>&1
