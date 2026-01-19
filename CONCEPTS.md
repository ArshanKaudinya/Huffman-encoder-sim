# Key Concepts & Learning Guide

This document explains the core concepts implemented in encoder-sim, designed for learning and study.

## 1. Canonical Huffman Coding

### What is Huffman Coding?

Huffman coding is a lossless compression algorithm that assigns variable-length bit codes to symbols based on frequency. More frequent symbols get shorter codes.

**Example:**
```
Input:  "AAAABBC"
Frequencies: A=4, B=2, C=1

Standard Huffman might assign:
  A: 0      (1 bit)
  B: 10     (2 bits)
  C: 11     (2 bits)

Encoded: 0 0 0 0 10 10 11 = "00001010​11" (10 bits vs 56 bits for ASCII)
```

### Why Canonical?

**Problem with standard Huffman:** You must transmit the entire tree structure to decode.

**Canonical Huffman solution:** Just transmit code *lengths* for each symbol. The decoder reconstructs codes using a canonical (standardized) assignment rule.

### Canonical Assignment Algorithm

Given code lengths for each symbol:

1. Sort symbols by (code_length, symbol_value)
2. Assign codes within each length sequentially
3. When moving to longer codes, left-shift and continue

**Example:**
```
Symbol  Length  Assigned Code
A       2       00
B       2       01
C       3       100
D       3       101
E       3       110

Rule: Codes of same length are sequential.
      Moving to length 3: (last_code_2 + 1) << 1 = (01 + 1) << 1 = 100
```

### Implementation Location

See [`crates/core/src/huffman.rs`](crates/core/src/huffman.rs)

**Key functions:**
- `build_code_lengths()`: Build Huffman tree using min-heap
- `canonicalize()`: Assign canonical codes from lengths
- `serialize_metadata()`: Export just the lengths
- `deserialize_metadata()`: Reconstruct codes from lengths

## 2. Bit-Level I/O

### Why Bit Operations?

Huffman codes are variable-length (not aligned to byte boundaries). We need to:
- Write individual bits into a byte stream
- Read individual bits from a byte stream
- Handle padding at byte boundaries

### MSB-First Order

The system uses **MSB-first** (most significant bit first) ordering:

```
Byte: 10110011
      ^       ^
      MSB     LSB

Reading left-to-right: bits 1, 0, 1, 1, 0, 0, 1, 1
```

This is standard for Huffman codes and network protocols.

### Padding Rules

When the bit stream doesn't end on a byte boundary, we pad with zeros:

```
Bits: 10111 (5 bits)
Padded byte: 10111000 (pad with 3 zeros)
```

The decoder must know how many symbols to decode (we store `raw_len` in the frame header).

### Implementation Location

See [`crates/core/src/bitio.rs`](crates/core/src/bitio.rs)

**Key structures:**
- `BitWriter`: Accumulates bits in a buffer, flushes complete bytes
- `BitReader`: Reads bits one-by-one from byte stream

## 3. Framing & Integrity

### What is a Frame?

A **frame** is a self-contained unit of data with:
- Header (metadata)
- Payload (actual data)
- Integrity check (CRC)

### Chunk Frame Structure

```
┌─────────────────────────────────┐
│ Header (26 bytes)               │
│  - Magic number                 │
│  - Chunk ID                     │
│  - Lengths                      │
│  - CRC32                        │
├─────────────────────────────────┤
│ Codec Metadata                  │
│  (Huffman codebook)             │
├─────────────────────────────────┤
│ Payload                         │
│  (compressed bits)              │
└─────────────────────────────────┘
```

### CRC32 (Cyclic Redundancy Check)

CRC is a checksum that detects corruption:

1. **Sender:** Compute CRC over data, include in header
2. **Receiver:** Recompute CRC, compare with header value
3. **Mismatch:** Data was corrupted, reject frame

**Properties:**
- Detects all single-bit errors
- Detects all double-bit errors
- Detects burst errors up to 32 bits
- Fast to compute (table-based)

### Implementation Location

See [`crates/core/src/framing.rs`](crates/core/src/framing.rs)

## 4. Packetization & Fragmentation

### Why Fragment?

Networks have a **Maximum Transmission Unit (MTU)**: the largest packet they can carry. Common MTUs:

- Ethernet: 1500 bytes
- Internet (IPv4): 576 bytes minimum
- Our system: 1200 bytes (default)

Large frames must be split into packets that fit the MTU.

### Packet Structure

```
┌─────────────────────────────────┐
│ Header (20 bytes)               │
│  - Magic number                 │
│  - Chunk ID                     │
│  - Packet ID (which fragment)   │
│  - Total Packets                │
│  - Fragment Length              │
├─────────────────────────────────┤
│ Fragment                        │
│  (slice of chunk frame)         │
└─────────────────────────────────┘
```

**Key idea:** Each packet knows:
- Which chunk it belongs to
- Its position in the sequence (packet_id)
- How many total packets exist

### Reassembly

Receiver collects all packets for a chunk_id, then concatenates fragments in order.

**Challenges:**
- Packets may arrive out of order
- Packets may be duplicated
- Packets may be lost

### Implementation Location

See [`crates/core/src/packet.rs`](crates/core/src/packet.rs) and [`crates/core/src/reassembly.rs`](crates/core/src/reassembly.rs)

## 5. Network Simulation

### Simulated Effects

1. **Latency:** Base delay for all packets
   - Models speed-of-light propagation + processing time

2. **Jitter:** Random variation in latency
   - Models queueing delays, routing variations

3. **Reordering:** Packets arrive out of order
   - Caused by jitter, multiple paths, queue priorities

4. **Loss:** Packets dropped randomly
   - Models congestion, corruption, routing failures

### Implementation Approach

Uses a **priority queue** (min-heap) keyed by delivery time:

1. When packet arrives: compute `delivery_time = now + latency + jitter`
2. Insert into heap with delivery_time
3. On receive: pop packets whose delivery_time has arrived

### Determinism via Seeding

All randomness uses a **seeded PRNG** (ChaCha8):

```rust
let mut rng = ChaCha8Rng::seed_from_u64(seed);
let jitter = rng.gen_range(0..jitter_ms);
```

Same seed → identical random sequence → reproducible results

### Implementation Location

See [`crates/core/src/network.rs`](crates/core/src/network.rs)

## 6. Bounded Memory & Flow Control

### The Problem

Without bounds, a fast sender can overwhelm a slow receiver:
- Sender generates packets faster than receiver processes them
- Unbounded queues grow indefinitely
- System runs out of memory

### Solution: Bounded Channels

Use channels with fixed capacity:

```rust
let (tx, rx) = bounded(32); // Max 32 items buffered

tx.send(item)?; // Blocks if channel is full (backpressure)
```

**Backpressure:** When receiver is slow, sender blocks, naturally slowing the whole pipeline.

### Reassembly Window

Track at most N chunks simultaneously:

```
Max memory = N chunks × packets_per_chunk × packet_size
Example: 64 chunks × 128 packets × 1200 bytes = ~10 MiB
```

If window is full and new chunk arrives, evict oldest incomplete chunk.

### Implementation Location

See pipeline in [`crates/app/src/main.rs`](crates/app/src/main.rs) and reassembler in [`crates/core/src/reassembly.rs`](crates/core/src/reassembly.rs)

## 7. Thread-Based Pipeline

### Architecture

```
[Chunker] → [Compressor] → [Packetizer] → [Network] → [Receiver] → [Decoder]
   thread       thread         thread       thread      thread       thread
              ↕ channel     ↕ channel    ↕ channel   ↕ channel
```

Each stage:
1. Receives data from input channel
2. Processes it
3. Sends result to output channel

### Benefits

- **Parallelism:** Stages run concurrently
- **Simplicity:** Each stage is independent, easy to reason about
- **Backpressure:** Bounded channels prevent runaway memory

### Graceful Shutdown

When a stage finishes:
1. Drop its sender (close output channel)
2. Downstream stage detects closed channel
3. Processes remaining items, then exits

### Implementation Location

See [`crates/app/src/main.rs`](crates/app/src/main.rs) - thread spawning in `run_transfer()`

## 8. Error Handling Philosophy

### No Panics

Every fallible operation returns `Result<T, Error>`:

```rust
pub fn parse_chunk_frame(bytes: &[u8]) -> Result<ChunkFrame> {
    if bytes.len() < HEADER_SIZE {
        return Err(FramingError::FrameTooShort { ... }.into());
    }
    // ...
}
```

**Benefits:**
- Caller can handle or propagate errors
- No sudden crashes
- Testable error conditions

### Structured Errors

Use `thiserror` for rich error types:

```rust
#[derive(Debug, Error)]
pub enum FramingError {
    #[error("invalid magic: expected {expected:?}, got {actual:?}")]
    InvalidMagic { expected: [u8; 4], actual: [u8; 4] },
    // ...
}
```

Errors carry context, making debugging easy.

### Implementation Location

See [`crates/core/src/error.rs`](crates/core/src/error.rs)

## 9. Testing Strategy

### Unit Tests

Test individual components in isolation:

```rust
#[test]
fn test_write_read_partial_bits() {
    let mut writer = BitWriter::new();
    writer.write_bits(0b101, 3).unwrap();

    let bytes = writer.finish();
    let mut reader = BitReader::new(&bytes);

    assert_eq!(reader.read_bits(3).unwrap(), 0b101);
}
```

Located at bottom of each module.

### Integration Tests

Test full pipeline end-to-end:

```rust
#[test]
fn test_full_pipeline_no_loss() {
    let input = b"test data";
    let frame = compress_and_frame(0, input).unwrap();
    let packets = packetize(0, &frame, mtu).unwrap();
    // ... send through network ...
    let decoded = decode(reassembled).unwrap();
    assert_eq!(decoded, input);
}
```

Located in [`tests/integration_test.rs`](tests/integration_test.rs)

### Property Testing Ideas

Future enhancements:
- **Round-trip property:** `decode(encode(data)) == data`
- **Determinism property:** `run(seed) == run(seed)`
- **Compression property:** `encoded.len() <= original.len()` (for repetitive data)

## 10. Systems Thinking

### Invariants

Properties that must always hold:

1. **Bit I/O:** `bit_count` in `BitWriter` is always < 8
2. **Reassembly:** At most `max_inflight` chunks tracked
3. **Channels:** Buffer size never exceeds capacity
4. **Packets:** `packet_id < total_packets` always

### Failure Modes

What can go wrong and how we handle it:

| Failure | Detection | Handling |
|---------|-----------|----------|
| Packet loss | Reassembly timeout | Error with chunk_id |
| Corruption | CRC mismatch | Reject frame, error |
| Reordering | Track packet_id sequence | Buffer and sort |
| Duplicate | Reassembler state | Detect and skip |
| Memory exhaustion | Bounded windows | Evict or block |

### Observability

Make system behavior visible:

- **Metrics:** Count events (packets sent, dropped, etc.)
- **Timings:** Measure durations (compression, transfer)
- **Logs:** Print errors and warnings
- **Config:** Show resolved parameters

This allows:
- Understanding what happened
- Debugging issues
- Reproducing runs
- Learning from behavior

## Further Reading

**Compression:**
- "Introduction to Data Compression" by Khalid Sayood
- RFC 1951 (DEFLATE, uses Huffman)

**Networking:**
- "Computer Networks" by Tanenbaum & Wetherall
- RFC 793 (TCP), RFC 9000 (QUIC)

**Systems Programming:**
- "The Rust Programming Language" book
- "Operating Systems: Three Easy Pieces" by Remzi

**Error Handling:**
- Rust Book Chapter 9: Error Handling
- "Error Handling in Rust" (blog posts by Burntsushi)

---

**Study Tips:**

1. Read code top-down (start with `main.rs`, follow the flow)
2. Run with `--print-config` to see what's happening
3. Modify parameters (e.g., `--loss 0.5`) and observe metrics
4. Add print statements to trace data flow
5. Break things intentionally (corrupt data, close channels) and see errors
6. Write your own tests for edge cases

**Questions to Explore:**

- What happens if MTU is very small (e.g., 50 bytes)?
- How does compression ratio change with different input patterns?
- What loss rate causes most runs to timeout?
- Can you modify the code to use async instead of threads?
- How would you add encryption to the pipeline?
