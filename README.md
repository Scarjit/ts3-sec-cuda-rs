# ts3-sec-cuda-rs

TeamSpeak 3 Security Level Tool - **GPU-accelerated** Rust implementation for improving TS3 identity security levels.

## Features

- 🔓 **Decode** identity files or strings
- ⚡ **GPU acceleration** via CUDA (NVIDIA GPUs)
- 🔧 **Tunable** kernel parameters for optimal performance
- 📊 **Benchmarking** suite included

Security levels are based on SHA-1 trailing zero bits, where level N proves ~2^N hashes were computed.

## Quick Start

```bash
# Build
make build

# Increase security level to 30 (GPU)
cargo run --release -- increase --file identity.ini --target 30 --method cuda

# Decode identity
cargo run --release -- decode --file identity.ini
```

## Installation

**Requirements:**
- Rust toolchain
- NVIDIA GPU + CUDA 13.0+ (optional, for GPU acceleration)

```bash
make build
# Or: cargo build --release
```

## Usage

### Increase Security Level (GPU - Recommended)

```bash
# From file (saves progress)
cargo run --release -- increase --file identity.ini --target 30 --method cuda

# From string (stdout only)
cargo run --release -- increase --string '14VHjz+...' --target 25 --method cuda

# Custom parameters
cargo run --release -- increase --file identity.ini --target 25 --method cuda \
  --cuda-threads 64 --batch-size 2000000
```

### Decode Identity

```bash
cargo run --release -- decode --file identity.ini
# Or: cargo run --release -- decode --string '14VHjz+...'
```

## Performance

### GPU (CUDA)
**Hardware:** NVIDIA GeForce RTX 4080 (Ada Lovelace)
**Performance:** 2.15 billion hashes/sec

| Level Range | Time (GPU) | Time (CPU)  |
|-------------|------------|-------------|
| 8 → 20      | ~1 second  | ~15 minutes |
| 8 → 25      | ~5 seconds | ~8 hours    |
| 8 → 30      | ~2 minutes | ~11 days    |

**Optimized defaults:**
- Threads per block: 128
- Batch size: 4,000,000
- Shared memory: 16,384 bytes (auto)
- Stack usage: 80 bytes

### CPU Fallback
**Hardware:** AMD Ryzen 9 5900X (Zen 3)
**Performance:** 15 million hashes/sec

## Options

```
--method <METHOD>              cpu (default) or cuda
--batch-size <SIZE>            CPU: 10k, CUDA: 4M (default)
--cuda-threads <THREADS>       32-1024 (default: 128)
--cuda-shared-mem <BYTES>      Auto-calculated by default
```

Run `cargo run -- increase --help` for all options.

## Benchmarking

Find optimal parameters for your GPU:
```bash
make bench
```

Tests thread counts (32/64/128/256) and batch sizes (50k-16M) for both short and long messages.

## Development

```bash
make build        # Build release binary
make test         # Run all tests (20 tests: CPU + CUDA)
make bench        # Run kernel parameter benchmarks
make nvcc-check   # Check CUDA stack usage (80 bytes)
```

## Technical Details

**CUDA Optimizations:**
- Dual-path kernel (fast <56 bytes, slow for TS3 keys ~109-113 bytes)
- Shared memory L1 caching (128 bytes static)
- Ring buffer (w[16] instead of w[80])
- Stack optimization (80 bytes, down from 112)
- Dynamic shared memory (128 bytes/thread for multi-block)

**Benchmark Results (RTX 4080):**

| Message Type | Threads | Batch Size | Throughput      |
|--------------|---------|------------|-----------------|
| Short (~16B) | 32      | 16M        | 3.88 GH/s       |
| Long (TS3)   | 128     | 4M         | **2.15 GH/s** ✓ |


---

**Identity Format:** `COUNTER || 'V' || base64(obfuscate(base64(KEYPAIR_ASN1)))`
**Security Level:** Trailing zero bits in `SHA-1(public_key || counter)`
**Key Type:** P-256 ECC (secp256r1)

## License

MIT License - see [LICENSE](LICENSE) file for details.
