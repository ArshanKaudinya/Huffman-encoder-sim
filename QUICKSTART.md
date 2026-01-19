# Quick Start Guide

## Prerequisites

Install Rust (if not already installed):
```bash
curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh
source $HOME/.cargo/env
```

## Build & Run

```bash
# Navigate to project
cd encoder-sim

# Build in release mode (optimized)
cargo build --release

# Run with defaults (generates sample data, random network)
cargo run --release

# You should see output like:
# Generating 262144 bytes of sample data (seed: 1705678934)...
# Starting transfer of 262144 bytes...
# Wrote 262144 bytes to "./out.bin"
#
# === Transfer Summary ===
# Duration: 1234 ms
# Input:  262144 bytes (0.25 MiB)
# Output: 262144 bytes (0.25 MiB)
# Verification: PASSED ✓
# ...
```

## Basic Examples

### 1. Deterministic Run (Reproducible)

```bash
cargo run --release -- --seed 42 --print-config
```

This prints the configuration and uses seed 42 for all randomness. Run it again with the same seed to get identical results.

### 2. Perfect Network (No Impairments)

```bash
cargo run --release -- --no-loss --latency 5 --jitter 0
```

Fast transfer with minimal latency and no packet loss.

### 3. Transfer a Real File

```bash
# Create a test file
echo "Hello, World!" > test.txt

# Transfer it
cargo run --release -- --in test.txt --out received.txt

# Verify
diff test.txt received.txt
```

### 4. Challenging Network Conditions

```bash
# High loss (will likely timeout and fail)
cargo run --release -- --loss 0.3 --latency 100
```

### 5. Large File Transfer

```bash
# Generate a 1MB test file
dd if=/dev/urandom of=large.bin bs=1M count=1

# Transfer with no loss
cargo run --release -- --in large.bin --out large_received.bin --no-loss

# Verify integrity
diff large.bin large_received.bin
echo "Files match!"
```

## Run Tests

```bash
# Run all tests (unit + integration)
cargo test

# Run with output visible
cargo test -- --nocapture

# Run only integration tests
cargo test --test integration_test

# Run specific test
cargo test test_full_pipeline_no_loss
```

## Understanding Output

### Configuration Section (with --print-config)

```
=== Configuration ===
Input file:  (generate sample)
Output file: "./out.bin"

Chunk size: 65536 bytes (64 KiB)
MTU: 1200 bytes

=== Network Simulation ===
Seed: 42
Base latency: 45 ms
Jitter: ±23 ms
Reorder window: 32 packets
Loss rate: 2.34%
```

### Transfer Summary

```
=== Transfer Summary ===
Duration: 523 ms              # Total time
Input:  262144 bytes          # Bytes read
Output: 262144 bytes          # Bytes written
Verification: PASSED ✓        # Output == Input?

=== Compression ===
Chunks: 4                     # Number of 64KB chunks
Compressed: 185671 (0.18 MiB) # Total compressed size
Ratio: 70.8%                  # Compression ratio (lower = better)

=== Network ===
Packets sent: 164             # Total packets generated
Packets dropped: 4 (2.44%)    # Lost due to simulation
Packets received: 160         # Successfully received
Packets reordered: 12 (7.50%) # Arrived out of order

=== Reassembly ===
Chunks reassembled: 4         # Chunks successfully reconstructed
Chunks decoded: 4             # Chunks decompressed
Timeouts: 0                   # Chunks that timed out
```

## Troubleshooting

### Build Errors

If you see compilation errors:
```bash
# Update Rust
rustup update

# Clean and rebuild
cargo clean
cargo build --release
```

### Test Failures

Some tests involve randomness. If a test fails:
```bash
# Run the specific test multiple times
cargo test test_name -- --nocapture

# Most tests use fixed seeds, so failures are bugs
# Please report them!
```

### Runtime Errors

**"chunks timed out":**
- Packet loss was too high for the timeout setting
- Try: `--no-loss` or increase `--timeout 5000`

**"channel closed":**
- A pipeline stage crashed
- Check for earlier error messages

**"CRC mismatch":**
- Data corruption detected (very rare with simulator)
- This is a bug - please report!

## Next Steps

1. **Read the Code:**
   - Start with [crates/app/src/main.rs](crates/app/src/main.rs) to see the pipeline
   - Read [crates/core/src/huffman.rs](crates/core/src/huffman.rs) to understand compression

2. **Experiment:**
   - Try different `--loss` rates and observe when transfers fail
   - Compare compression ratios with different input patterns
   - Modify chunk size (`--chunk-bytes`) and see effect on packet count

3. **Learn More:**
   - Read [CONCEPTS.md](CONCEPTS.md) for detailed explanations
   - Read [README.md](README.md) for complete documentation

4. **Modify:**
   - Add new metrics (e.g., track retransmissions)
   - Implement an alternative codec (e.g., RLE)
   - Add encryption layer between compressor and network

## Common Command Patterns

```bash
# Quick test with known seed
cargo run --release -- --seed 42

# Silent mode (only show result)
cargo run --release -- --no-metrics --seed 42

# Verbose deterministic run
cargo run --release -- --seed 42 --print-config

# Benchmark large transfer
time cargo run --release -- --in large_file.bin --no-loss --no-metrics

# Extreme conditions
cargo run --release -- --loss 0.1 --latency 200 --jitter 100 --timeout 5000
```

## IDE Setup

### VSCode

Install extensions:
- `rust-analyzer` (Rust language support)
- `CodeLLDB` (debugger)

Open workspace:
```bash
code encoder-sim
```

### IntelliJ / CLion

Open `encoder-sim` folder, IntelliJ will detect it as a Rust project.

## Performance Notes

**Debug vs Release:**
- Debug builds (`cargo run`) are ~10x slower
- Always use `cargo run --release` for realistic performance

**Expected Performance:**
- ~0.2-2 MB/s throughput (depends on network simulation settings)
- Compression takes ~50% of total time
- Network simulation adds significant overhead (for realism)

## Getting Help

1. **Documentation:** Read [README.md](README.md) and [CONCEPTS.md](CONCEPTS.md)
2. **Code Comments:** All modules have explanatory comments
3. **Tests:** Look at tests for usage examples
4. **Issues:** Open a GitHub issue (if this were a real project)

---

**Happy learning!** 🦀
