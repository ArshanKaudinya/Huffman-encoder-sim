# encoder-sim

**Educational file transfer system with chunked canonical Huffman compression over simulated unreliable networks.**

This is a system demonstrating:
- **Compression**: Canonical Huffman coding with deterministic code assignment
- **Framing**: Chunk serialization with CRC integrity checks
- **Packetization**: MTU-aware fragmentation
- **Network simulation**: Latency, jitter, reordering, and packet loss
- **Reassembly**: Bounded-memory packet collection with timeout handling
- **Pipeline architecture**: Thread-based concurrency with backpressure

## Features

✓ **No panics**: All errors are structured and handled gracefully
✓ **Bounded memory**: Channels, reassembly windows, and buffers are all fixed-size
✓ **Deterministic**: Seeded randomness makes runs fully reproducible
✓ **Observable**: Comprehensive metrics for understanding system behavior
✓ **Tested**: Unit tests for each module + integration tests for full pipeline

## Quick Start

```bash
# Build the project
cargo build --release

# Run with default settings (generates sample data, random network params)
cargo run --release

# Deterministic run with seed
cargo run --release -- --seed 42

# Transfer a specific file
cargo run --release -- --in input.txt --out output.txt

# Perfect network (no loss, minimal latency)
cargo run --release -- --no-loss --latency 5

# Print configuration before running
cargo run --release -- --print-config --seed 123

# Run tests
cargo test
```

## Architecture

```
Input File
    ↓
┌─────────────────┐
│ Chunker Thread  │  Read file in fixed-size chunks
└────────┬────────┘
         ↓ [bounded channel]
┌─────────────────┐
│Compressor Thread│  Huffman encode each chunk + frame with metadata
└────────┬────────┘
         ↓ [bounded channel]
┌─────────────────┐
│Packetizer Thread│  Fragment frames into MTU-sized packets
└────────┬────────┘
         ↓ [bounded channel]
┌─────────────────┐
│ Network Thread  │  Simulate latency, jitter, reordering, loss
└────────┬────────┘
         ↓ [bounded channel]
┌─────────────────┐
│Receiver Thread  │  Reassemble packets into chunks (bounded window)
└────────┬────────┘
         ↓ [bounded channel]
┌─────────────────┐
│ Decoder Thread  │  Decompress chunks and write to output
└────────┬────────┘
         ↓
Output File
```

## Binary Formats

### Chunk Frame Format

```
Offset  Field               Size    Description
------  -----               ----    -----------
0       Magic               4       0x48464654 ("HFFT")
4       chunk_id            8       u64 chunk identifier
12      raw_len             4       u32 original uncompressed size
16      codec_meta_len      2       u16 Huffman codebook size
18      payload_len         4       u32 compressed payload size
22      crc32               4       u32 integrity checksum
26      codec_metadata      var     Huffman codebook (symbol lengths)
...     payload             var     Compressed bits

Total: 26 + codec_meta_len + payload_len bytes
```

### Packet Format

```
Offset  Field               Size    Description
------  -----               ----    -----------
0       Magic               2       0x504B ("PK")
2       chunk_id            8       u64 which chunk this belongs to
10      packet_id           4       u32 fragment index (0-based)
14      total_packets       4       u32 total fragments for chunk
18      fragment_len        2       u16 bytes in fragment
20      fragment            var     Slice of chunk frame

Total: 20 + fragment_len bytes (must be ≤ MTU)
```

## Canonical Huffman Codec

The system uses **canonical Huffman coding**, which has two key properties:

1. **Optimal compression**: Code lengths are computed using Huffman's algorithm (min-heap/priority queue)
2. **Canonical codes**: Within each code length, symbols are assigned codes in ascending order

### Why Canonical?

Standard Huffman requires transmitting the full tree structure. Canonical Huffman only requires transmitting *code lengths* for each symbol, making the metadata much more compact. The receiver can reconstruct the exact codes from just the lengths.

### Algorithm

**Encoding:**
1. Count symbol frequencies in input chunk
2. Build optimal code lengths (Huffman tree)
3. Assign canonical codes: sort by (length, symbol), assign sequential codes per length
4. Encode input using the codebook
5. Serialize codebook as (symbol, length) pairs

**Decoding:**
1. Deserialize codebook metadata
2. Reconstruct canonical codes from lengths
3. Decode bit stream using lookup/traversal

### Determinism

Tie-breaking: when two symbols have equal frequency, the smaller symbol value gets the smaller code. This ensures bit-identical output given the same input.

## Network Simulation

The network simulator applies:

- **Latency**: Base delay for all packets
- **Jitter**: Random variation (uniform distribution)
- **Reordering**: Packets may arrive out of order within a window
- **Loss**: Packets dropped with probability `loss_rate`

All randomness is seeded for reproducibility.

### Default Behavior

When run without arguments, the tool generates random network parameters but **prints them**, so you can reproduce the run by specifying the seed:

```bash
$ cargo run
# Outputs:
# Seed: 1234567890
# Base latency: 45 ms
# Jitter: ±23 ms
# Loss rate: 0.02%
# ...

# To reproduce:
$ cargo run -- --seed 1234567890
```

## Configuration Options

| Flag                  | Description                           | Default              |
|-----------------------|---------------------------------------|----------------------|
| `--in <PATH>`         | Input file                            | Generate sample      |
| `--out <PATH>`        | Output file                           | `./out.bin`          |
| `--seed <N>`          | Random seed for determinism           | Current timestamp    |
| `--chunk-bytes <N>`   | Chunk size in bytes                   | 65536 (64 KiB)       |
| `--mtu <N>`           | Maximum packet size                   | 1200                 |
| `--latency <MS>`      | Base network latency                  | Random 10-80         |
| `--jitter <MS>`       | Latency jitter range                  | Random 0-40          |
| `--reorder-window <N>`| Packet reorder window                 | Random 0-64          |
| `--loss <RATE>`       | Packet loss rate (0.0-1.0)            | Random 0-0.05        |
| `--no-loss`           | Disable packet loss                   | -                    |
| `--timeout <MS>`      | Reassembly timeout                    | Random 500-3000      |
| `--max-inflight <N>`  | Max chunks in flight                  | 64                   |
| `--channel-capacity <N>` | Pipeline channel capacity           | 32                   |
| `--print-config`      | Print resolved configuration          | false                |
| `--no-metrics`        | Don't print detailed metrics          | false                |
| `--help, -h`          | Print help message                    | -                    |

## Failure Modes

### Packet Loss

**Default mode**: System fails fast when packets are lost
- Reassembler times out waiting for missing packets
- Returns clear error with chunk_id and missing packet count
- Pipeline shuts down cleanly

**No-loss mode** (`--no-loss`): Loss rate = 0.0, guaranteed success

### Corruption

- CRC mismatch after reassembly: error with expected/actual CRC values
- Malformed packet header: packet skipped, metric incremented
- Invalid Huffman code during decode: structured error

### Bounded Memory

- Max 64 chunks in flight (configurable)
- If reassembler window is full, oldest incomplete chunk is evicted (warning)
- All channels have fixed capacity (default 32)

## Metrics

After each run, the tool prints:

```
=== Transfer Summary ===
Duration: 1234 ms
Input:  262144 bytes (0.25 MiB)
Output: 262144 bytes (0.25 MiB)
Verification: PASSED ✓

=== Compression ===
Chunks: 4
Raw bytes: 262144 (0.25 MiB)
Compressed: 195432 (0.19 MiB)
Ratio: 74.5%

=== Network ===
Packets generated: 164
Packets sent: 164
Packets dropped: 3 (1.83%)
Packets received: 161
Packets reordered: 12 (7.45%)
Packets duplicate: 0
Packets invalid: 0

=== Reassembly ===
Chunks reassembled: 4
Chunks decoded: 4
CRC failures: 0
Timeouts: 0

=== Performance ===
Throughput: 0.21 MB/s
```

## Project Structure

```
encoder-sim/
├── Cargo.toml              # Workspace root
├── crates/
│   ├── core/               # Library crate
│   │   ├── src/
│   │   │   ├── lib.rs
│   │   │   ├── bitio.rs        # Bit-level I/O
│   │   │   ├── huffman.rs      # Canonical Huffman codec
│   │   │   ├── framing.rs      # Chunk serialization
│   │   │   ├── packet.rs       # Packetization
│   │   │   ├── reassembly.rs   # Packet reassembly
│   │   │   ├── network.rs      # Network simulator
│   │   │   ├── metrics.rs      # Metrics collection
│   │   │   └── error.rs        # Error types
│   │   └── Cargo.toml
│   └── app/                # Binary crate
│       ├── src/
│       │   ├── main.rs         # CLI + pipeline orchestration
│       │   ├── config.rs       # Config parsing
│       │   └── input_gen.rs    # Sample data generation
│       └── Cargo.toml
├── tests/
│   └── integration_test.rs # End-to-end tests
└── README.md
```

## Learning Goals

This project teaches:

1. **Data encoding**: Variable-length codes, canonical forms, metadata serialization
2. **Framing**: How to package data for transmission with integrity checks
3. **Packetization**: MTU constraints, fragmentation, reassembly
4. **Network simulation**: Latency models, packet loss, reordering
5. **Flow control**: Bounded channels, backpressure, windowing
6. **Concurrency**: Thread-based pipeline with message passing
7. **Error handling**: Structured errors, graceful shutdown, no panics
8. **Testing**: Unit tests, integration tests, deterministic reproduction
9. **Systems thinking**: Memory bounds, observable behavior, failure modes

## Non-Goals

This is **not**:
- A production compressor (use zstd, lz4, etc.)
- A production network protocol (use TCP, QUIC, etc.)
- Optimized for speed (use async, zero-copy, SIMD, etc.)

The focus is **correctness, clarity, and learning**.

## Dependencies

Minimal, justified dependencies:

- `crossbeam-channel`: Bounded channels for backpressure
- `rand`, `rand_chacha`: Seeded randomness for determinism
- `crc32fast`: Fast CRC32 computation
- `thiserror`: Ergonomic error handling

## Testing

```bash
# Run all tests
cargo test

# Run with output
cargo test -- --nocapture

# Run specific test
cargo test test_full_pipeline_no_loss

# Run integration tests only
cargo test --test integration_test
```

## Examples

### Deterministic Transfer

```bash
cargo run --release -- --seed 42 --print-config
```

### High Loss, High Latency

```bash
# Will likely fail due to timeouts
cargo run --release -- --loss 0.2 --latency 100 --timeout 500
```

### Large File, No Loss

```bash
# Generate a 10 MB file
dd if=/dev/urandom of=large.bin bs=1M count=10

cargo run --release -- --in large.bin --out received.bin --no-loss

# Verify
diff large.bin received.bin
```

## Contributing

This is an educational project. If you find bugs or have suggestions for making it more instructive, please open an issue!

## License

MIT

---

**Built with Rust for learning systems programming, compression, and network protocols.**
