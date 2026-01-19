# Complete Guide to encoder-sim

**A comprehensive technical deep-dive into building an educational file transfer system with compression and network simulation.**

---

## Table of Contents

1. [Introduction](#1-introduction)
2. [Prerequisites & Foundations](#2-prerequisites--foundations)
3. [System Architecture](#3-system-architecture)
4. [Core Algorithms](#4-core-algorithms)
5. [Binary Formats](#5-binary-formats)
6. [Implementation Details](#6-implementation-details)
7. [Concurrency & Threading](#7-concurrency--threading)
8. [Error Handling](#8-error-handling)
9. [Testing Strategy](#9-testing-strategy)
10. [Performance Considerations](#10-performance-considerations)
11. [Reproducing This Project](#11-reproducing-this-project)

---

## 1. Introduction

### 1.1 What is encoder-sim?

encoder-sim is an **educational file transfer system** that demonstrates:
- How compression works (Huffman coding)
- How data is transmitted over networks (packetization)
- How unreliable networks behave (loss, reordering, latency)
- How to build resilient systems (reassembly, timeouts, integrity checks)

### 1.2 Learning Objectives

By studying this project, you will understand:
- Information theory and lossless compression
- Binary protocol design
- Network behavior and simulation
- Concurrent programming patterns
- Systems programming in Rust

### 1.3 Project Goals

**Educational principles:**
- Every design decision has a clear rationale
- No "magic" - all behavior is deterministic and reproducible
- Observable - comprehensive metrics show what's happening
- Safe - no panics, all errors are structured and recoverable

**Non-goals:**
- Production performance (we prioritize clarity over speed)
- Maximum compression ratio (we use Huffman, not modern codecs)
- Real networking (we simulate, not actually send packets over TCP/UDP)

---

## 2. Prerequisites & Foundations

### 2.1 Information Theory Basics

#### 2.1.1 Entropy

**Entropy** measures the "information content" of a message. It's the average number of bits needed to encode each symbol.

**Formula:**
```
H(X) = -Σ p(x) * log₂(p(x))
```

Where:
- `p(x)` = probability of symbol x
- `log₂` = logarithm base 2

**Example:**
- Message: "AAABBC"
- Frequencies: A=3, B=2, C=1
- Probabilities: P(A)=3/6=0.5, P(B)=2/6=0.33, P(C)=1/6=0.17
- Entropy: -(0.5*log₂(0.5) + 0.33*log₂(0.33) + 0.17*log₂(0.17)) ≈ 1.46 bits

This means we need ~1.46 bits on average per symbol (better than 8 bits for ASCII!).

#### 2.1.2 Prefix-Free Codes

A **prefix-free code** ensures no code is a prefix of another code.

**Example of prefix-free (valid):**
```
A = 0
B = 10
C = 11
```

**Example of NOT prefix-free (invalid):**
```
A = 0
B = 01   ← "0" is a prefix of "01"
C = 11
```

Why is this important? With prefix-free codes, you can decode unambiguously:
- Bits: `01011`
- Decoding: `0|10|11` = A, B, C (no ambiguity!)

### 2.2 Huffman Coding

#### 2.2.1 The Algorithm

Huffman coding creates an **optimal prefix-free code** based on symbol frequencies.

**Steps:**
1. Count frequency of each symbol
2. Create leaf nodes for each symbol with its frequency
3. Build tree bottom-up:
   - Take two nodes with lowest frequency
   - Create parent node with combined frequency
   - Repeat until one node remains (root)
4. Assign codes: left edge = 0, right edge = 1
5. Code for each symbol = path from root to leaf

**Example:**

Frequencies: A=5, B=2, C=1, D=1

```
Step 1: Create leaves
  [A,5]  [B,2]  [C,1]  [D,1]

Step 2: Combine C and D (lowest frequencies: 1+1=2)
  [A,5]  [B,2]  [CD,2]
                 /    \
               [C,1]  [D,1]

Step 3: Combine B and CD (both have frequency 2, total=4)
  [A,5]  [BCD,4]
         /      \
      [B,2]    [CD,2]
               /    \
             [C,1]  [D,1]

Step 4: Combine A and BCD (5+4=9)
      [ABCD,9]
      /       \
   [A,5]    [BCD,4]
            /      \
         [B,2]    [CD,2]
                  /    \
                [C,1]  [D,1]

Step 5: Assign codes (left=0, right=1)
  A: 0        (1 bit)
  B: 10       (2 bits)
  C: 110      (3 bits)
  D: 111      (3 bits)
```

**Verification:**
- Average bits = (5×1 + 2×2 + 1×3 + 1×3) / 9 = 15/9 = 1.67 bits per symbol
- Original (8-bit): 9 symbols × 8 bits = 72 bits
- Huffman: 5+4+3+3 = 15 bits
- Compression ratio: 15/72 = 20.8% (79.2% saved!)

#### 2.2.2 Why Huffman is Optimal

**Proof sketch:**
1. Huffman assigns shorter codes to more frequent symbols
2. The two least frequent symbols are siblings at maximum depth
3. This minimizes the weighted average code length
4. No other prefix-free code can do better

### 2.3 Canonical Huffman Coding

Standard Huffman has a problem: **you need to transmit the entire tree structure** to the decoder.

**Canonical Huffman** solves this by normalizing codes:

**Rules:**
1. Codes of the same length are assigned in **ascending symbol order**
2. The first code of length L is the value: `(last_code_of_length_L-1 + 1) << 1`

**Example:**

Standard Huffman might give:
```
A (freq=5): 0
B (freq=2): 10
C (freq=1): 110
D (freq=1): 111
```

Canonical Huffman with same **lengths** (1, 2, 3, 3):
```
Step 1: Sort by (length, symbol):
  A(1), B(2), C(3), D(3)

Step 2: Assign codes sequentially:
  A: 0        (length 1, first code)
  B: 10       (length 2, shift previous: (0+1)<<1 = 10)
  C: 110      (length 3, shift previous: (10+1)<<1 = 110)
  D: 111      (length 3, next code: 110+1 = 111)
```

**Key insight:** You only need to transmit the **code lengths**, not the actual codes! The decoder can reconstruct codes using the canonical algorithm.

**Metadata format:**
```
Number of symbols: 4
(A, length=1)
(B, length=2)
(C, length=3)
(D, length=3)
```

This is much more compact than transmitting the tree structure.

### 2.4 Network Fundamentals

#### 2.4.1 MTU (Maximum Transmission Unit)

**MTU** = largest packet size that can be transmitted in one piece.

**Common MTUs:**
- Ethernet: 1500 bytes
- Internet path MTU: typically 1280-1500 bytes
- Jumbo frames: 9000 bytes

**Why it matters:**
If your data is larger than MTU, it must be **fragmented** into multiple packets.

**Example:**
- Data: 5000 bytes
- MTU: 1200 bytes
- Required packets: ceil(5000/1200) = 5 packets
  - Packet 0: bytes 0-1199
  - Packet 1: bytes 1200-2399
  - Packet 2: bytes 2400-3599
  - Packet 3: bytes 3600-4799
  - Packet 4: bytes 4800-4999

#### 2.4.2 Network Impairments

Real networks are **unreliable**:

**1. Latency** - Delay between send and receive
- Example: 50ms from New York to London
- Caused by: speed of light, routing, processing

**2. Jitter** - Variation in latency
- Example: packets arrive at 50ms, 53ms, 48ms, 55ms
- Caused by: varying queue lengths, route changes

**3. Reordering** - Packets arrive out of order
- Example: send [A, B, C], receive [A, C, B]
- Caused by: different routes, parallel processing

**4. Loss** - Packets never arrive
- Example: send 100 packets, receive 98
- Caused by: buffer overflow, corruption, routing errors

#### 2.4.3 Reassembly Challenges

When packets arrive out of order or with loss, reassembly is hard:

**Scenario 1: Out of order**
```
Send: [Pkt0, Pkt1, Pkt2, Pkt3]
Recv: [Pkt0, Pkt2, Pkt1, Pkt3]
Action: Buffer Pkt2, wait for Pkt1, then emit in order
```

**Scenario 2: Loss**
```
Send: [Pkt0, Pkt1, Pkt2, Pkt3]
Recv: [Pkt0, Pkt2, Pkt3] (Pkt1 lost!)
Action: Wait for Pkt1... timeout after 1 second... declare failure
```

**Scenario 3: Delayed packet**
```
Send: [Pkt0, Pkt1, Pkt2, Pkt3]
Recv: [Pkt0, Pkt2, Pkt3] ... wait ... wait ... [Pkt1 arrives!]
Action: Successfully reassemble
```

### 2.5 Concurrency Patterns

#### 2.5.1 Pipeline Parallelism

A **pipeline** breaks work into stages that run concurrently:

```
Input → [Stage 1] → [Stage 2] → [Stage 3] → Output
         Thread 1     Thread 2     Thread 3
```

**Example: encoder-sim pipeline**
```
File → [Compress] → [Packetize] → [Network] → [Reassemble] → [Decompress] → Output
        Thread 1      Thread 2      Thread 3     Thread 4       Main Thread
```

**Benefits:**
- Each stage processes different data simultaneously
- Better CPU utilization
- Natural separation of concerns

#### 2.5.2 Message Passing with Channels

**Channels** = thread-safe queues for sending data between threads

**Types:**
1. **Unbounded** - Can grow infinitely (dangerous!)
2. **Bounded** - Fixed capacity (provides backpressure)

**Backpressure example:**
```rust
let (tx, rx) = bounded(10); // capacity = 10

// Producer
for i in 0..100 {
    tx.send(i); // Blocks when buffer is full!
}

// Consumer
while let Ok(item) = rx.recv() {
    process(item);
}
```

When the buffer is full, `tx.send()` blocks, preventing the producer from overwhelming the consumer.

#### 2.5.3 Shared State with Mutex

**Mutex** = Mutual Exclusion lock

```rust
let counter = Arc::new(Mutex::new(0));

// Thread 1
{
    let mut count = counter.lock().unwrap();
    *count += 1;
} // Lock released here

// Thread 2
{
    let mut count = counter.lock().unwrap();
    *count += 1;
} // Lock released here
```

**Arc** = Atomic Reference Counted pointer (allows sharing across threads)

---

## 3. System Architecture

### 3.1 High-Level Overview

```
┌─────────────────────────────────────────────────────────────┐
│                         INPUT DATA                           │
│                    (File or Generated)                       │
└────────────────┬────────────────────────────────────────────┘
                 │
                 ▼
┌────────────────────────────────────────────────────────────┐
│  STAGE 1: CHUNKER/COMPRESSOR                               │
│  - Split data into fixed-size chunks                       │
│  - Build Huffman codebook for each chunk                   │
│  - Compress using Huffman                                  │
│  - Serialize to binary frame format                        │
└────────────────┬───────────────────────────────────────────┘
                 │ Chunk Frames
                 ▼
┌────────────────────────────────────────────────────────────┐
│  STAGE 2: PACKETIZER                                       │
│  - Fragment frames into MTU-sized packets                  │
│  - Add packet headers (chunk_id, packet_id, etc.)          │
└────────────────┬───────────────────────────────────────────┘
                 │ Packets
                 ▼
┌────────────────────────────────────────────────────────────┐
│  STAGE 3: NETWORK SIMULATOR                                │
│  - Add latency (delay each packet)                         │
│  - Add jitter (randomize delay)                            │
│  - Reorder packets (shuffle within window)                 │
│  - Drop packets (random loss)                              │
└────────────────┬───────────────────────────────────────────┘
                 │ Impaired Packets
                 ▼
┌────────────────────────────────────────────────────────────┐
│  STAGE 4: RECEIVER/REASSEMBLER                             │
│  - Collect packets for each chunk                          │
│  - Detect missing/duplicate packets                        │
│  - Reassemble into chunk frames (in order!)                │
│  - Handle timeouts for incomplete chunks                   │
└────────────────┬───────────────────────────────────────────┘
                 │ Chunk Frames
                 ▼
┌────────────────────────────────────────────────────────────┐
│  STAGE 5: DECODER/WRITER                                   │
│  - Parse chunk frame headers                               │
│  - Validate CRC checksums                                  │
│  - Decompress using Huffman                                │
│  - Write to output file                                    │
└────────────────┬───────────────────────────────────────────┘
                 │
                 ▼
┌────────────────────────────────────────────────────────────┐
│                       OUTPUT FILE                           │
└────────────────────────────────────────────────────────────┘
```

### 3.2 Module Structure

**Workspace layout:**
```
encoder-sim/
├── Cargo.toml                    # Workspace root
├── crates/
│   ├── core/                     # Library (reusable logic)
│   │   ├── src/
│   │   │   ├── lib.rs           # Module exports
│   │   │   ├── error.rs         # Error types
│   │   │   ├── bitio.rs         # Bit-level I/O
│   │   │   ├── huffman.rs       # Huffman codec
│   │   │   ├── framing.rs       # Chunk serialization
│   │   │   ├── packet.rs        # Packetization
│   │   │   ├── network.rs       # Network simulation
│   │   │   ├── reassembly.rs    # Packet reassembly
│   │   │   └── metrics.rs       # Observability
│   │   └── tests/
│   │       └── integration_test.rs
│   └── app/                      # Binary (main application)
│       └── src/
│           ├── main.rs          # Pipeline orchestration
│           ├── config.rs        # CLI argument parsing
│           └── input_gen.rs     # Test data generation
└── target/                       # Build artifacts
```

**Design rationale:**
- **core** = pure library, no I/O, fully testable
- **app** = thin orchestration layer, handles files and CLI

### 3.3 Data Flow

**Type transformations:**

```rust
Vec<u8>                          // Raw input data
  ↓ (split into chunks)
Vec<u8>                          // Single chunk (e.g., 64 KiB)
  ↓ (compress_and_frame)
Vec<u8>                          // Chunk frame (binary format)
  ↓ (packetize)
Vec<Packet>                      // Packets (MTU-sized)
  ↓ (network simulator)
Vec<Packet>                      // Impaired packets (reordered/dropped)
  ↓ (reassemble)
Vec<u8>                          // Chunk frame (reconstructed)
  ↓ (decompress_frame)
Vec<u8>                          // Decompressed chunk
  ↓ (concatenate all chunks)
Vec<u8>                          // Complete output
```

### 3.4 Thread Model

**5 threads + main:**

```
[Main Thread]
  ├─ Load input
  ├─ Spawn worker threads
  ├─ Run decoder (blocking on channel)
  └─ Join all threads

[Thread 1: Chunker]
  ├─ Read from: input_data (owned)
  ├─ Write to: chunk_tx (bounded channel)
  └─ Task: Split and compress chunks

[Thread 2: Packetizer]
  ├─ Read from: chunk_rx
  ├─ Write to: packet_tx
  └─ Task: Fragment into packets

[Thread 3: Network]
  ├─ Read from: packet_rx
  ├─ Write to: network_tx
  └─ Task: Simulate network impairments

[Thread 4: Receiver]
  ├─ Read from: network_rx
  ├─ Write to: reassembled_tx
  └─ Task: Reassemble packets into chunks

[Main Thread: Decoder]
  ├─ Read from: reassembled_rx
  ├─ Write to: output_file
  └─ Task: Decompress and write
```

**Channel capacities:**
All channels have bounded capacity (default: 32) to provide backpressure.

### 3.5 Error Propagation

**Strategy: Fail fast, propagate up**

```rust
// Each thread returns Result<()>
chunker_thread(...) -> Result<()>

// Errors propagate via ?
let frame = compress_and_frame(chunk_id, chunk)?;

// Main thread collects results
chunker_thread.join()??;  // First ? = thread panic, second ? = Result
```

**Error sources:**
1. I/O errors (file not found, permission denied)
2. Protocol errors (invalid magic, CRC mismatch)
3. Resource errors (channel closed, window full)
4. Timeout errors (chunk reassembly timeout)

---

## 4. Core Algorithms

### 4.1 Bit-Level I/O

#### 4.1.1 Why Bit I/O?

Huffman codes are **variable length** - some codes are 3 bits, others 7 bits, etc. You can't just write bytes; you need bit-level control.

#### 4.1.2 BitWriter Implementation

**Concept:** Accumulate bits in a buffer, flush when full.

```rust
pub struct BitWriter {
    bytes: Vec<u8>,      // Completed bytes
    bit_buffer: u8,      // Current byte being filled
    bit_count: u8,       // How many bits in bit_buffer (0-7)
}
```

**Writing bits (MSB-first):**

```
Example: Write 5 bits: 10110

State initially:
  bit_buffer = 00000000
  bit_count = 0

Write 10110 (5 bits):
  1. Extract top 5 bits of value: 10110
  2. Shift left to align: 10110000 (shift by 8-5=3)
  3. OR into buffer: 10110000
  4. bit_count = 5

State after:
  bit_buffer = 10110000
  bit_count = 5

Write 111 (3 bits):
  1. Extract: 111
  2. Shift to fill remaining space: 00000111 (shift by 8-5-3=0)
  3. OR into buffer: 10110111
  4. bit_count = 8 (buffer full!)
  5. Flush: bytes.push(10110111)
  6. Reset: bit_buffer = 0, bit_count = 0
```

**Padding:**
When finished, if `bit_count > 0`, we have partial byte. Pad with zeros on the right:
```
bit_buffer = 10110000
bit_count = 5
→ Flush as-is (already padded with zeros)
```

**Code:**
```rust
pub fn write_bits(&mut self, value: u64, count: usize) -> Result<()> {
    let mut val = value;
    let mut remaining = count;

    while remaining > 0 {
        // How many bits fit in current byte?
        let bits_to_write = remaining.min(8 - self.bit_count as usize);

        // Extract top bits from value
        let shift = remaining - bits_to_write;
        let bits = ((val >> shift) & ((1 << bits_to_write) - 1)) as u8;

        // Add to buffer (shifted to align)
        self.bit_buffer |= bits << (8 - self.bit_count as usize - bits_to_write);
        self.bit_count += bits_to_write as u8;

        // Flush if full
        if self.bit_count == 8 {
            self.bytes.push(self.bit_buffer);
            self.bit_buffer = 0;
            self.bit_count = 0;
        }

        // Remove written bits from value
        val &= (1 << shift) - 1;
        remaining -= bits_to_write;
    }

    Ok(())
}
```

#### 4.1.3 BitReader Implementation

**Concept:** Read bits one at a time from byte array.

```rust
pub struct BitReader<'a> {
    bytes: &'a [u8],     // Source bytes
    byte_pos: usize,     // Current byte index
    bit_pos: u8,         // Current bit in byte (0-7, left to right)
}
```

**Reading bits (MSB-first):**

```
Bytes: [10110111, 00101010, ...]
       ↑
       byte_pos=0, bit_pos=0

Read 5 bits:
  1. Read bits 0-4 from byte 0: 10110
  2. byte_pos=0, bit_pos=5

Read 4 bits:
  1. Read bits 5-7 from byte 0: 111 (3 bits)
  2. Read bits 0-0 from byte 1: 0 (1 bit)
  3. Result: 1110
  4. byte_pos=1, bit_pos=1
```

**Code:**
```rust
pub fn read_bits(&mut self, count: usize) -> Result<u64> {
    let mut result = 0u64;
    let mut remaining = count;

    while remaining > 0 {
        if self.byte_pos >= self.bytes.len() {
            return Err(BitIoError::EndOfStream.into());
        }

        // How many bits available in current byte?
        let bits_available = 8 - self.bit_pos as usize;
        let bits_to_read = remaining.min(bits_available);

        // Extract bits from current byte
        let byte = self.bytes[self.byte_pos];
        let shift = bits_available - bits_to_read;
        let mask = (1 << bits_to_read) - 1;
        let bits = ((byte >> shift) & mask) as u64;

        // Add to result
        result = (result << bits_to_read) | bits;

        // Update position
        self.bit_pos += bits_to_read as u8;
        if self.bit_pos >= 8 {
            self.byte_pos += 1;
            self.bit_pos = 0;
        }

        remaining -= bits_to_read;
    }

    Ok(result)
}
```

### 4.2 Huffman Codec

#### 4.2.1 Building the Tree

**Input:** Frequency array (256 elements, one per byte value)

**Algorithm:**
```rust
fn build_code_lengths(freqs: &[u64; 256], active: &[u8]) -> Result<[u8; 256]> {
    // 1. Create nodes for active symbols
    struct Node {
        freq: u64,
        min_symbol: u8,          // For tie-breaking
        left: Option<Box<Node>>,
        right: Option<Box<Node>>,
        symbol: Option<u8>,      // Some(...) for leaves
    }

    // 2. Build min-heap
    let mut heap = BinaryHeap::new();
    for &sym in active {
        heap.push(Node {
            freq: freqs[sym],
            min_symbol: sym,
            left: None,
            right: None,
            symbol: Some(sym),
        });
    }

    // 3. Merge until one node remains
    while heap.len() > 1 {
        let left = heap.pop().unwrap();
        let right = heap.pop().unwrap();

        heap.push(Node {
            freq: left.freq + right.freq,
            min_symbol: left.min_symbol.min(right.min_symbol),
            left: Some(Box::new(left)),
            right: Some(Box::new(right)),
            symbol: None,
        });
    }

    // 4. Traverse tree to compute depths
    let root = heap.pop().unwrap();
    let mut lengths = [0u8; 256];

    fn compute_depths(node: &Node, depth: usize, lengths: &mut [u8; 256]) {
        if let Some(sym) = node.symbol {
            lengths[sym as usize] = depth as u8;
        } else {
            if let Some(ref left) = node.left {
                compute_depths(left, depth + 1, lengths);
            }
            if let Some(ref right) = node.right {
                compute_depths(right, depth + 1, lengths);
            }
        }
    }

    compute_depths(&root, 0, &mut lengths);
    Ok(lengths)
}
```

**Key detail: Tie-breaking**
When two nodes have the same frequency, we break ties by `min_symbol`. This ensures deterministic builds.

#### 4.2.2 Canonicalization

**Input:** Code lengths array
**Output:** Canonical codes

```rust
fn canonicalize(lengths: &[u8; 256]) -> ([u32; 256], Vec<(u8, u8)>) {
    let mut codes = [0u32; 256];

    // 1. Collect (length, symbol) pairs and sort
    let mut symbols_by_length: Vec<_> = (0..256)
        .filter(|&i| lengths[i] > 0)
        .map(|i| (lengths[i], i as u8))
        .collect();
    symbols_by_length.sort_unstable();

    // 2. Assign codes
    let mut code = 0u32;
    let mut prev_length = 0u8;

    for &(length, symbol) in &symbols_by_length {
        // When length increases, shift left
        if length > prev_length {
            code <<= length - prev_length;
            prev_length = length;
        }

        codes[symbol as usize] = code;
        code += 1;
    }

    (codes, symbols_by_length)
}
```

**Example:**
```
Lengths: A=2, B=2, C=3, D=3

Step 1: Sort by (length, symbol):
  [(2, A), (2, B), (3, C), (3, D)]

Step 2: Assign codes:
  A: code=00, length=2
     code=0, assign to A, increment code=1

  B: code=01, length=2
     (same length, no shift)
     code=1, assign to B, increment code=2

  C: code=100, length=3
     (length increased: 2→3, shift left: 2<<1=4=100)
     code=4, assign to C, increment code=5

  D: code=101, length=3
     (same length, no shift)
     code=5, assign to D

Result:
  A = 00
  B = 01
  C = 100
  D = 101
```

#### 4.2.3 Encoding

```rust
pub fn encode(&self, data: &[u8]) -> Result<Vec<u8>> {
    let mut writer = BitWriter::new();

    for &symbol in data {
        let length = self.lengths[symbol as usize];
        if length == 0 {
            return Err(HuffmanError::InvalidCode { position: 0 }.into());
        }

        let code = self.codes[symbol as usize];
        writer.write_bits(code as u64, length as usize)?;
    }

    Ok(writer.finish())
}
```

**Example:**
```
Input: "AB"
Codebook: A=0 (1 bit), B=10 (2 bits)

Encode A: write_bits(0, 1) → buffer = 0_______
Encode B: write_bits(2, 2) → buffer = 010_____

Finish: pad with zeros → 01000000 = 0x40
```

#### 4.2.4 Decoding

```rust
pub fn decode(&self, bits: &[u8], expected_len: usize) -> Result<Vec<u8>> {
    let mut reader = BitReader::new(bits);
    let mut result = Vec::with_capacity(expected_len);

    while result.len() < expected_len {
        let mut code = 0u32;
        let mut length = 0usize;

        // Read bits until we find a matching code
        loop {
            let bit = reader.read_bit()?;
            code = (code << 1) | (bit as u32);
            length += 1;

            if let Some(symbol) = self.find_symbol(code, length as u8) {
                result.push(symbol);
                break;
            }

            if length > MAX_CODE_LENGTH {
                return Err(HuffmanError::InvalidCode {
                    position: reader.position()
                }.into());
            }
        }
    }

    Ok(result)
}

fn find_symbol(&self, code: u32, length: u8) -> Option<u8> {
    for i in 0..256 {
        if self.lengths[i] == length && self.codes[i] == code {
            return Some(i as u8);
        }
    }
    None
}
```

**Example:**
```
Bits: 01000000
Codebook: A=0 (len 1), B=10 (len 2)

Decode:
  Read bit: 0, code=0, length=1
  Check: find_symbol(0, 1) → Some(A) ✓
  Output: A

  Read bit: 1, code=1, length=1
  Check: find_symbol(1, 1) → None
  Read bit: 0, code=10, length=2
  Check: find_symbol(10, 2) → Some(B) ✓
  Output: B

Result: "AB"
```

### 4.3 CRC32 Checksums

#### 4.3.1 What is CRC?

**CRC** = Cyclic Redundancy Check

It's a hash function that detects errors in data:
- Single-bit errors: 100% detection
- Two-bit errors: 100% detection
- Burst errors: very high detection

**How it works (simplified):**
1. Treat data as a big polynomial in GF(2)
2. Divide by a generator polynomial
3. The remainder is the CRC

**Example (CRC-8 for illustration):**
```
Data: 11010011
Generator: 1011 (x³ + x + 1)

Division in GF(2):
  11010011 ÷ 1011 = remainder: 100

CRC = 100
```

In practice, we use CRC32 with a standard polynomial, and the `crc32fast` crate does the math.

#### 4.3.2 Usage in encoder-sim

```rust
use crc32fast::Hasher;

fn compute_crc(chunk_id: u64, raw_len: u32, meta: &[u8], payload: &[u8]) -> u32 {
    let mut hasher = Hasher::new();
    hasher.update(&chunk_id.to_le_bytes());
    hasher.update(&raw_len.to_le_bytes());
    hasher.update(&meta.len().to_le_bytes());
    hasher.update(&payload.len().to_le_bytes());
    hasher.update(meta);
    hasher.update(payload);
    hasher.finalize()
}
```

**Validation:**
```rust
fn validate_crc(frame: &[u8]) -> Result<()> {
    let stored_crc = parse_crc_from_header(frame);
    let computed_crc = compute_crc(frame);

    if stored_crc != computed_crc {
        return Err(FramingError::CrcMismatch {
            expected: stored_crc,
            actual: computed_crc
        }.into());
    }

    Ok(())
}
```

### 4.4 Network Simulation

#### 4.4.1 Priority Queue for Time-Based Events

**Concept:** Store packets with their delivery time, pop in chronological order.

```rust
struct DelayedPacket {
    packet: Packet,
    delivery_time: Instant,
}

impl Ord for DelayedPacket {
    fn cmp(&self, other: &Self) -> Ordering {
        // Min-heap: earlier time = higher priority
        other.delivery_time.cmp(&self.delivery_time)
    }
}

pub struct NetworkSimulator {
    queue: BinaryHeap<DelayedPacket>,
    rng: ChaCha8Rng,
    config: NetworkConfig,
}
```

#### 4.4.2 Sending Packets

```rust
pub fn send(&mut self, packet: Packet) {
    // 1. Decide if we drop this packet
    if self.rng.gen::<f64>() < self.config.loss_rate {
        // Drop!
        return;
    }

    // 2. Compute delay
    let base_delay = self.config.base_latency_ms;
    let jitter = self.rng.gen_range(0..=self.config.jitter_ms);
    let total_delay = Duration::from_millis(base_delay + jitter);

    // 3. Compute delivery time
    let delivery_time = Instant::now() + total_delay;

    // 4. Add to priority queue
    self.queue.push(DelayedPacket {
        packet,
        delivery_time,
    });
}
```

#### 4.4.3 Receiving Packets

```rust
pub fn recv(&mut self) -> Option<Packet> {
    // Check if any packet is ready
    if let Some(delayed) = self.queue.peek() {
        if Instant::now() >= delayed.delivery_time {
            // Ready! Pop and return
            return Some(self.queue.pop().unwrap().packet);
        }
    }

    None  // No packet ready yet
}
```

#### 4.4.4 Reordering

Reordering happens naturally with latency + jitter:

```
Send Pkt0 at t=0, delay=50ms → deliver at t=50
Send Pkt1 at t=1, delay=30ms → deliver at t=31  ← Pkt1 arrives first!
Send Pkt2 at t=2, delay=60ms → deliver at t=62
```

We can also explicitly shuffle within a window:

```rust
let mut window = Vec::new();
for _ in 0..config.reorder_window {
    if let Some(pkt) = self.queue.pop() {
        window.push(pkt);
    }
}

// Shuffle
window.shuffle(&mut self.rng);

// Put back
for pkt in window {
    self.queue.push(pkt);
}
```

### 4.5 Packet Reassembly

#### 4.5.1 Data Structures

```rust
struct ChunkState {
    total_packets: u32,
    packets: HashMap<u32, Packet>,  // packet_id → packet
    first_seen: Instant,
}

pub struct Reassembler {
    chunks: HashMap<u64, ChunkState>,           // chunk_id → state
    completed_chunks: HashMap<u64, Vec<u8>>,   // chunk_id → frame bytes
    next_expected_chunk: u64,
    max_inflight: usize,
    timeout_ms: u64,
}
```

#### 4.5.2 Inserting Packets

```rust
pub fn insert_packet(&mut self, packet: Packet) -> Result<Option<(u64, Vec<u8>)>> {
    let chunk_id = packet.chunk_id;

    // 1. Check if already completed
    if self.completed_chunks.contains_key(&chunk_id) {
        return Ok(None);  // Ignore duplicate
    }

    // 2. Check if too old
    if chunk_id < self.next_expected_chunk {
        return Ok(None);  // Already emitted
    }

    // 3. Get or create chunk state
    if !self.chunks.contains_key(&chunk_id) {
        if self.chunks.len() >= self.max_inflight {
            return Err(ReassemblyError::WindowFull {
                max: self.max_inflight
            }.into());
        }
        self.chunks.insert(chunk_id, ChunkState::new(packet.total_packets));
    }

    // 4. Insert packet
    let chunk_state = self.chunks.get_mut(&chunk_id).unwrap();

    if chunk_state.packets.contains_key(&packet.packet_id) {
        return Err(ReassemblyError::DuplicatePacket {
            chunk_id,
            packet_id: packet.packet_id,
        }.into());
    }

    chunk_state.packets.insert(packet.packet_id, packet);

    // 5. Check if complete
    if chunk_state.packets.len() == chunk_state.total_packets as usize {
        let chunk_state = self.chunks.remove(&chunk_id).unwrap();
        let mut packets: Vec<_> = chunk_state.packets.into_values().collect();
        let frame_bytes = reassemble(&mut packets)?;

        self.completed_chunks.insert(chunk_id, frame_bytes);
        return Ok(self.try_emit_next());
    }

    Ok(None)
}
```

#### 4.5.3 In-Order Delivery

```rust
fn try_emit_next(&mut self) -> Option<(u64, Vec<u8>)> {
    // Only emit if next expected chunk is ready
    if let Some(bytes) = self.completed_chunks.remove(&self.next_expected_chunk) {
        let chunk_id = self.next_expected_chunk;
        self.next_expected_chunk += 1;
        Some((chunk_id, bytes))
    } else {
        None
    }
}

pub fn drain_completed(&mut self) -> Vec<(u64, Vec<u8>)> {
    let mut result = Vec::new();

    // Keep emitting while consecutive chunks are ready
    while let Some(chunk) = self.try_emit_next() {
        result.push(chunk);
    }

    result
}
```

**Example:**
```
Receive chunk 2: buffer it (waiting for chunk 0)
Receive chunk 1: buffer it
Receive chunk 0: emit 0, then drain → emit 1, emit 2
```

#### 4.5.4 Timeout Handling

```rust
pub fn check_timeouts(&mut self) -> Vec<ReassemblyError> {
    let now = Instant::now();
    let mut errors = Vec::new();
    let mut timed_out = Vec::new();

    for (&chunk_id, state) in &self.chunks {
        let elapsed = now.duration_since(state.first_seen);
        if elapsed > Duration::from_millis(self.timeout_ms) {
            errors.push(ReassemblyError::Timeout {
                chunk_id,
                packets_received: state.packets.len(),
                packets_expected: state.total_packets as usize,
            });
            timed_out.push(chunk_id);
        }
    }

    // Remove timed-out chunks
    for chunk_id in timed_out {
        self.chunks.remove(&chunk_id);
    }

    errors
}
```

---

## 5. Binary Formats

### 5.1 Chunk Frame Format

**Total: 26-byte header + variable metadata + variable payload**

```
Offset  Size  Field             Type    Description
------  ----  ----------------  ------  ---------------------------
0       4     magic             u32     0x464D4843 ("CHMF" ASCII)
4       8     chunk_id          u64     Chunk sequence number
12      4     raw_len           u32     Uncompressed data size
16      2     codec_meta_len    u16     Huffman metadata size
18      4     payload_len       u32     Compressed bits size
22      4     crc32             u32     Checksum
26      ?     codec_metadata    bytes   Huffman codebook
?       ?     payload           bytes   Compressed data
```

**All multi-byte integers are little-endian.**

**Example frame:**
```
Magic:          43 48 4D 46              (CHMF)
Chunk ID:       00 00 00 00 00 00 00 00  (0)
Raw length:     0D 00 00 00              (13 bytes)
Meta length:    06 00                    (6 bytes)
Payload length: 08 00 00 00              (8 bytes)
CRC32:          A7 B2 C3 D4              (example)
--- Header ends (26 bytes) ---
Metadata:       03 00 41 01 42 02 43 03  (3 symbols: A=1, B=2, C=3)
Payload:        F8 A0 ...                (compressed bits)
```

### 5.2 Packet Format

**Total: 20-byte header + variable fragment**

```
Offset  Size  Field             Type    Description
------  ----  ----------------  ------  ---------------------------
0       2     magic             u16     0x504B ("PK" ASCII)
2       8     chunk_id          u64     Which chunk this belongs to
10      4     packet_id         u32     Sequence within chunk (0-based)
14      4     total_packets     u32     Total packets for this chunk
18      2     fragment_len      u16     Size of fragment in this packet
20      ?     fragment          bytes   Piece of chunk frame
```

**Reassembly:**
1. Sort packets by `packet_id`
2. Concatenate fragments in order
3. Result = original chunk frame

**Example:**
```
Chunk frame size: 5000 bytes
MTU: 1200 bytes
Packets needed: ceil(5000/1200) = 5

Packet 0: header (20 bytes) + fragment (1180 bytes) = 1200 bytes
  chunk_id = 0
  packet_id = 0
  total_packets = 5
  fragment_len = 1180
  fragment = bytes[0:1180]

Packet 1: header (20 bytes) + fragment (1180 bytes) = 1200 bytes
  chunk_id = 0
  packet_id = 1
  total_packets = 5
  fragment_len = 1180
  fragment = bytes[1180:2360]

...

Packet 4: header (20 bytes) + fragment (200 bytes) = 220 bytes
  chunk_id = 0
  packet_id = 4
  total_packets = 5
  fragment_len = 200
  fragment = bytes[4800:5000]
```

### 5.3 Huffman Metadata Format

```
Offset  Size  Field             Description
------  ----  ----------------  ------------------------
0       2     num_symbols       u16, count of active symbols
2       ?     symbol_entries    (symbol, length) pairs

Symbol entry (2 bytes):
  0     1     symbol            u8, the byte value (0-255)
  1     1     code_length       u8, bits in code (1-255)
```

**Example:**
```
Symbols: A=2 bits, B=2 bits, C=3 bits

Metadata:
  03 00        num_symbols = 3
  41 02        symbol 'A' (0x41), length 2
  42 02        symbol 'B' (0x42), length 2
  43 03        symbol 'C' (0x43), length 3

Total size: 2 + 3*2 = 8 bytes
```

**Decoding:** Receiver uses lengths to reconstruct canonical codes.

---

## 6. Implementation Details

### 6.1 Error Handling

#### 6.1.1 Error Type Hierarchy

```rust
// Top-level error type
#[derive(Debug, thiserror::Error)]
pub enum Error {
    #[error("bit I/O error: {0}")]
    BitIo(#[from] BitIoError),

    #[error("Huffman error: {0}")]
    Huffman(#[from] HuffmanError),

    #[error("framing error: {0}")]
    Framing(#[from] FramingError),

    #[error("packet error: {0}")]
    Packet(#[from] PacketError),

    #[error("reassembly error: {0}")]
    Reassembly(#[from] ReassemblyError),

    #[error("I/O error: {0}")]
    Io(#[from] std::io::Error),

    #[error("channel error: {0}")]
    Channel(String),
}

// Specific error types
#[derive(Debug, thiserror::Error)]
pub enum HuffmanError {
    #[error("empty frequency table")]
    EmptyFrequencyTable,

    #[error("invalid code at position {position}")]
    InvalidCode { position: usize },

    #[error("code length too long: {length}")]
    CodeLengthTooLong { length: usize },

    #[error("length mismatch: expected {expected}, got {actual}")]
    LengthMismatch { expected: usize, actual: usize },
}
```

#### 6.1.2 Error Propagation Pattern

```rust
// Using ? operator
fn process_chunk(chunk: &[u8]) -> Result<Vec<u8>> {
    let frame = compress_and_frame(0, chunk)?;  // Can fail
    let packets = packetize(0, &frame, 1200)?;  // Can fail
    let reassembled = reassemble(&mut packets)?;  // Can fail
    Ok(reassembled)
}

// Error bubbles up automatically
match process_chunk(data) {
    Ok(result) => println!("Success!"),
    Err(e) => eprintln!("Error: {}", e),
}
```

#### 6.1.3 No Panics Policy

**Bad (panics):**
```rust
let byte = data[100];  // Panics if data.len() < 101
```

**Good (returns error):**
```rust
let byte = data.get(100).ok_or(Error::OutOfBounds)?;
```

### 6.2 Memory Management

#### 6.2.1 Bounded Buffers

All channels have fixed capacity:
```rust
let (tx, rx) = bounded(32);  // Max 32 items in flight
```

All data structures have limits:
```rust
pub struct Reassembler {
    max_inflight: usize,  // Max 64 chunks at once
    // ...
}
```

#### 6.2.2 Move Semantics

Data is moved (not copied) between pipeline stages:

```rust
// Chunker produces chunk frame
let frame: Vec<u8> = compress_and_frame(id, chunk)?;

// Send through channel (moves ownership)
tx.send(frame)?;  // frame moved into channel

// Receiver gets ownership
let frame = rx.recv()?;  // frame moved out of channel

// No copying! Same allocation throughout.
```

#### 6.2.3 Pre-allocation

When output size is known, pre-allocate:

```rust
let mut result = Vec::with_capacity(expected_len);
// Avoids multiple reallocations
```

### 6.3 Determinism

#### 6.3.1 Seeded Randomness

All randomness uses a seeded RNG:

```rust
use rand::SeedableRng;
use rand_chacha::ChaCha8Rng;

let mut rng = ChaCha8Rng::seed_from_u64(42);

// Deterministic sequence
println!("{}", rng.gen::<f64>());  // Always same value for seed 42
```

#### 6.3.2 Huffman Tie-Breaking

When symbols have equal frequency, sort by symbol value:

```rust
impl Ord for Node {
    fn cmp(&self, other: &Self) -> Ordering {
        (other.freq, other.min_symbol)
            .cmp(&(self.freq, self.min_symbol))
    }
}
```

This ensures the same Huffman tree every time.

### 6.4 Metrics Collection

#### 6.4.1 Shared Metrics Object

```rust
let metrics = Arc::new(Mutex::new(Metrics::new()));

// Each thread gets a clone
let metrics_clone = Arc::clone(&metrics);

// Update from any thread
{
    let mut m = metrics.lock().unwrap();
    m.packets_sent += 1;
}
```

#### 6.4.2 Metrics Fields

```rust
pub struct Metrics {
    // Input/output
    pub input_bytes: u64,
    pub output_bytes: u64,

    // Compression
    pub chunks_created: u64,
    pub raw_chunk_bytes: u64,
    pub compressed_chunk_bytes: u64,

    // Network
    pub packets_generated: u64,
    pub packets_sent: u64,
    pub packets_dropped: u64,
    pub packets_received: u64,
    pub packets_reordered: u64,

    // Reassembly
    pub chunks_reassembled: u64,
    pub chunks_decoded: u64,
    pub chunks_timed_out: u64,

    // Timing
    pub start_time: Instant,
    pub end_time: Option<Instant>,
}
```

#### 6.4.3 Reorder Tracking

```rust
pub struct ReorderTracker {
    last_packet_id: HashMap<u64, u32>,  // chunk_id → last packet_id seen
}

pub fn track(&mut self, chunk_id: u64, packet_id: u32) -> bool {
    let last = self.last_packet_id.entry(chunk_id).or_insert(0);

    let is_reordered = packet_id < *last;
    *last = (*last).max(packet_id);

    is_reordered
}
```

---

## 7. Concurrency & Threading

### 7.1 Channel Communication

#### 7.1.1 Channel Types

**Bounded channel:**
```rust
use crossbeam_channel::bounded;

let (tx, rx) = bounded(10);  // Capacity: 10

// Sender blocks when full
for i in 0..100 {
    tx.send(i).unwrap();  // Blocks when 10 items in channel
}
```

**Unbounded channel:**
```rust
use crossbeam_channel::unbounded;

let (tx, rx) = unbounded();

// Never blocks (can cause OOM!)
for i in 0..1_000_000 {
    tx.send(i).unwrap();  // Always succeeds
}
```

#### 7.1.2 Backpressure

Bounded channels provide **natural backpressure**:

```
[Fast Producer] --send()--> [Channel: 10 items] --recv()--> [Slow Consumer]
                             ↑ Full!

Producer blocks on send() until consumer drains channel.
```

### 7.2 Thread Lifecycle

#### 7.2.1 Spawning

```rust
let handle = std::thread::spawn(move || {
    // Thread code here
    process_data();

    // Return Result
    Ok(())
});
```

#### 7.2.2 Joining

```rust
// Wait for thread to finish
let result = handle.join();

match result {
    Ok(Ok(())) => println!("Thread succeeded"),
    Ok(Err(e)) => println!("Thread returned error: {}", e),
    Err(_) => println!("Thread panicked!"),
}

// Shorthand with ??
handle.join()??;  // First ? = panic, second ? = error
```

#### 7.2.3 Dropping Channels to Signal Completion

```rust
// Producer
{
    let (tx, rx) = bounded(10);

    for i in 0..100 {
        tx.send(i).unwrap();
    }

    drop(tx);  // Signal end of stream
}

// Consumer
while let Ok(item) = rx.recv() {
    process(item);
}
// Loop ends when all senders dropped
```

### 7.3 Synchronization Patterns

#### 7.3.1 Mutex for Shared State

```rust
let counter = Arc::new(Mutex::new(0));

let handles: Vec<_> = (0..10)
    .map(|_| {
        let counter = Arc::clone(&counter);
        std::thread::spawn(move || {
            for _ in 0..100 {
                let mut c = counter.lock().unwrap();
                *c += 1;
            } // Lock released here
        })
    })
    .collect();

for handle in handles {
    handle.join().unwrap();
}

println!("Counter: {}", counter.lock().unwrap());  // 1000
```

#### 7.3.2 Lock Scoping

**Bad (lock held too long):**
```rust
let mut data = shared.lock().unwrap();
process(&data);      // Lock held during processing
network_call(&data); // Lock held during I/O!
*data += 1;
```

**Good (minimal lock scope):**
```rust
let copy = {
    let data = shared.lock().unwrap();
    data.clone()
}; // Lock released here

process(&copy);
network_call(&copy);

{
    let mut data = shared.lock().unwrap();
    *data += 1;
} // Lock released here
```

### 7.4 Pipeline Pattern

```
Thread 1 produces → Channel A → Thread 2 consumes/produces → Channel B → Thread 3
```

**Benefits:**
- Each stage can run at its own pace
- Bounded channels prevent memory explosion
- Easy to add/remove stages
- Natural separation of concerns

**Implementation:**
```rust
let (tx1, rx1) = bounded(32);
let (tx2, rx2) = bounded(32);

let t1 = thread::spawn(move || {
    for i in 0..100 {
        tx1.send(i).unwrap();
    }
});

let t2 = thread::spawn(move || {
    while let Ok(x) = rx1.recv() {
        tx2.send(x * 2).unwrap();
    }
});

let t3 = thread::spawn(move || {
    while let Ok(x) = rx2.recv() {
        println!("{}", x);
    }
});

t1.join().unwrap();
t2.join().unwrap();
t3.join().unwrap();
```

---

## 8. Error Handling

### 8.1 Error Design Philosophy

**Principles:**
1. **No panics** - All errors are catchable
2. **Structured errors** - Use enums, not strings
3. **Context** - Include relevant information
4. **Propagation** - Use `?` operator
5. **Recovery** - Caller decides what to do

### 8.2 Error Types

#### 8.2.1 Using thiserror

```rust
use thiserror::Error;

#[derive(Debug, Error)]
pub enum MyError {
    #[error("file not found: {path}")]
    FileNotFound { path: String },

    #[error("invalid magic number: expected {expected:#x}, got {actual:#x}")]
    InvalidMagic { expected: u32, actual: u32 },

    #[error("timeout after {duration:?}")]
    Timeout { duration: Duration },
}
```

**Benefits:**
- Automatic `Display` impl
- Automatic `Error` trait impl
- Nice error messages

#### 8.2.2 Error Context

```rust
#[error("CRC mismatch: expected {expected:#x}, got {actual:#x}")]
CrcMismatch { expected: u32, actual: u32 },
```

When this error occurs:
```
Error: CRC mismatch: expected 0xa7b2c3d4, got 0x12345678
```

Much better than:
```
Error: CRC failed
```

### 8.3 Result Type

```rust
pub type Result<T> = std::result::Result<T, Error>;
```

**Usage:**
```rust
fn may_fail() -> Result<u32> {
    if some_condition {
        Ok(42)
    } else {
        Err(Error::SomethingWrong)
    }
}

// Caller handles error
match may_fail() {
    Ok(value) => println!("Got: {}", value),
    Err(e) => eprintln!("Error: {}", e),
}

// Or propagate with ?
fn caller() -> Result<u32> {
    let value = may_fail()?;
    Ok(value * 2)
}
```

### 8.4 Error Conversion

#### 8.4.1 From Trait

```rust
#[derive(Debug, Error)]
pub enum Error {
    #[error("Huffman error: {0}")]
    Huffman(#[from] HuffmanError),  // Auto-converts HuffmanError
}
```

**Usage:**
```rust
fn process() -> Result<()> {
    let codebook = Codebook::from_frequencies(&freqs)?;
    // If from_frequencies returns Err(HuffmanError),
    // it's automatically converted to Error::Huffman
    Ok(())
}
```

#### 8.4.2 Manual Conversion

```rust
tx.send(packet)
    .map_err(|_| Error::Channel("sender channel closed".to_string()))?;
```

---

## 9. Testing Strategy

### 9.1 Unit Tests

#### 9.1.1 Test Organization

```rust
// In the same file as code
#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_basic_functionality() {
        let result = my_function(42);
        assert_eq!(result, 84);
    }
}
```

#### 9.1.2 Example: BitIO Tests

```rust
#[test]
fn test_write_read_round_trip() {
    let mut writer = BitWriter::new();
    writer.write_bits(0b10110, 5).unwrap();
    writer.write_bits(0b111, 3).unwrap();
    let bytes = writer.finish();

    let mut reader = BitReader::new(&bytes);
    assert_eq!(reader.read_bits(5).unwrap(), 0b10110);
    assert_eq!(reader.read_bits(3).unwrap(), 0b111);
}

#[test]
fn test_padding() {
    let mut writer = BitWriter::new();
    writer.write_bits(0b101, 3).unwrap();
    let bytes = writer.finish();

    // 3 bits → padded to 8 bits: 10100000
    assert_eq!(bytes, vec![0b10100000]);
}
```

#### 9.1.3 Example: Huffman Tests

```rust
#[test]
fn test_encode_decode_round_trip() {
    let data = b"hello world";

    // Build codebook
    let mut freqs = [0u64; 256];
    for &byte in data {
        freqs[byte as usize] += 1;
    }
    let codebook = Codebook::from_frequencies(&freqs).unwrap();

    // Encode
    let encoded = codebook.encode(data).unwrap();

    // Decode
    let decoded = codebook.decode(&encoded, data.len()).unwrap();

    assert_eq!(decoded, data);
}

#[test]
fn test_determinism() {
    let mut freqs = [0u64; 256];
    freqs[b'A' as usize] = 5;
    freqs[b'B' as usize] = 5;  // Same frequency!

    let codebook1 = Codebook::from_frequencies(&freqs).unwrap();
    let codebook2 = Codebook::from_frequencies(&freqs).unwrap();

    // Same codes every time
    assert_eq!(codebook1.codes, codebook2.codes);
}
```

### 9.2 Integration Tests

#### 9.2.1 Test Location

```
crates/core/tests/integration_test.rs
```

#### 9.2.2 Full Pipeline Test

```rust
#[test]
fn test_full_pipeline_no_loss() {
    let input_data = b"test data...";

    // Compress
    let frame = compress_and_frame(0, input_data).unwrap();

    // Packetize
    let packets = packetize(0, &frame, 100).unwrap();

    // Network (perfect)
    let config = NetworkConfig::perfect(42);
    let mut network = NetworkSimulator::new(config);
    for pkt in packets {
        network.send(pkt);
    }

    // Receive
    let mut received = Vec::new();
    while let Some(pkt) = network.recv() {
        received.push(pkt);
    }

    // Reassemble
    let reassembled = reassemble(&mut received).unwrap();

    // Decompress
    let frame = parse_chunk_frame(&reassembled).unwrap();
    let decoded = decompress_frame(&frame).unwrap();

    assert_eq!(decoded, input_data);
}
```

### 9.3 Property-Based Testing Ideas

**Idea 1: Round-trip property**
```
∀ data: encode(decode(data)) == data
```

**Idea 2: Compression ratio**
```
∀ data: compressed_size(data) ≤ original_size(data) + overhead
```

**Idea 3: Network invariants**
```
∀ packets: (received ⊆ sent) ∧ (no_duplicates(received))
```

### 9.4 Test Coverage

**What to test:**
- Happy path (normal operation)
- Edge cases (empty input, single byte, max size)
- Error conditions (invalid magic, CRC mismatch)
- Boundary conditions (MTU exact fit, MTU overflow)
- Concurrency (race conditions, deadlocks)

**Example:**
```rust
#[test]
fn test_empty_data() { /* ... */ }

#[test]
fn test_single_byte() { /* ... */ }

#[test]
fn test_max_size() { /* ... */ }

#[test]
fn test_invalid_magic() { /* ... */ }

#[test]
fn test_crc_mismatch() { /* ... */ }
```

---

## 10. Performance Considerations

### 10.1 Compression Performance

**Time complexity:**
- Building Huffman tree: O(n log n) where n = number of unique symbols
- Encoding: O(m) where m = input size
- Decoding: O(m × k) where k = average code length

**Space complexity:**
- Codebook: O(1) - fixed size (256 symbols)
- Encoded data: O(m)

**Optimization opportunities:**
- Use lookup table for decoding (instead of linear search)
- Vectorize bit operations
- Parallel chunk compression

### 10.2 Network Simulation Performance

**Priority queue:**
- Insert: O(log n)
- Pop min: O(log n)

**For 10,000 packets:**
- Total: 10,000 × log(10,000) ≈ 133,000 operations

**Optimization:**
- Use specialized data structure (calendar queue)
- Batch processing

### 10.3 Memory Usage

**Per-chunk overhead:**
```
Chunk state: ~100 bytes
Packets: n × (20 byte header + fragment)
Total: O(chunk_size + packet_count)
```

**Bounded memory:**
- Max chunks in flight: 64
- Max channel size: 32 items
- Total bounded by configuration

### 10.4 Thread Contention

**Mutex contention:**
```rust
// Bad: frequent locking
for i in 0..1000 {
    metrics.lock().unwrap().counter += 1;
}

// Good: batch updates
let local_count = 1000;
metrics.lock().unwrap().counter += local_count;
```

**Lock-free alternatives:**
```rust
use std::sync::atomic::{AtomicU64, Ordering};

let counter = Arc::new(AtomicU64::new(0));
counter.fetch_add(1, Ordering::Relaxed);
```

---

## 11. Reproducing This Project

### 11.1 Step-by-Step Build Guide

#### Step 1: Create Workspace

```bash
mkdir encoder-sim
cd encoder-sim
```

Create `Cargo.toml`:
```toml
[workspace]
resolver = "2"
members = [
    "crates/core",
    "crates/app",
]

[workspace.dependencies]
crossbeam-channel = "0.5"
rand = "0.8"
rand_chacha = "0.3"
crc32fast = "1.4"
thiserror = "1.0"

[profile.release]
opt-level = 3
lto = true
```

#### Step 2: Create Core Library

```bash
mkdir -p crates/core/src
cd crates/core
```

Create `Cargo.toml`:
```toml
[package]
name = "encoder-sim-core"
version = "0.1.0"
edition = "2021"

[dependencies]
thiserror = { workspace = true }
crc32fast = { workspace = true }
crossbeam-channel = { workspace = true }
rand = { workspace = true }
rand_chacha = { workspace = true }
```

Create `src/lib.rs`:
```rust
pub mod error;
pub mod bitio;
pub mod huffman;
pub mod framing;
pub mod packet;
pub mod network;
pub mod reassembly;
pub mod metrics;

pub use error::{Error, Result};
```

#### Step 3: Implement Each Module

**Follow this order (dependencies first):**

1. `error.rs` - Define error types
2. `bitio.rs` - Implement BitWriter and BitReader
3. `huffman.rs` - Implement Huffman codec
4. `framing.rs` - Implement chunk framing
5. `packet.rs` - Implement packetization
6. `network.rs` - Implement network simulator
7. `reassembly.rs` - Implement packet reassembly
8. `metrics.rs` - Implement metrics tracking

**For each module:**
- Write the data structures
- Implement core logic
- Add unit tests
- Run `cargo test`

#### Step 4: Create Application Binary

```bash
cd ../..
mkdir -p crates/app/src
cd crates/app
```

Create `Cargo.toml`:
```toml
[package]
name = "encoder-sim"
version = "0.1.0"
edition = "2021"

[[bin]]
name = "encoder-sim"
path = "src/main.rs"

[dependencies]
encoder-sim-core = { path = "../core" }
crossbeam-channel = { workspace = true }
rand = { workspace = true }
rand_chacha = { workspace = true }
```

Create `src/` files:
- `config.rs` - CLI parsing
- `input_gen.rs` - Test data generation
- `main.rs` - Pipeline orchestration

#### Step 5: Write Integration Tests

Create `crates/core/tests/integration_test.rs`:
```rust
use encoder_sim_core::*;

#[test]
fn test_full_pipeline() {
    // Test complete flow
}
```

#### Step 6: Documentation

Create:
- `README.md` - Project overview
- `CONCEPTS.md` - Algorithm details
- `QUICKSTART.md` - Getting started guide

#### Step 7: Build and Run

```bash
cargo build --release
cargo test
./target/release/encoder-sim --seed 42 --no-loss
```

### 11.2 Key Implementation Decisions

#### Decision 1: Why Canonical Huffman?

**Alternatives considered:**
- Standard Huffman: needs to transmit tree structure
- LZ77: more complex, harder to teach
- Arithmetic coding: patented, complex

**Chosen: Canonical Huffman**
- Compact metadata (only lengths needed)
- Easy to understand
- Deterministic
- Good compression for educational purposes

#### Decision 2: Why Thread-Based Pipeline?

**Alternatives considered:**
- Async/await: more complex, harder to debug
- Single-threaded: no concurrency benefits
- Actor model: overkill for this use case

**Chosen: Thread-based pipeline**
- Simple mental model
- Natural parallelism
- Standard library support
- Easy to debug

#### Decision 3: Why Bounded Channels?

**Alternatives considered:**
- Unbounded channels: risk OOM
- Custom queue: reinventing the wheel

**Chosen: Bounded channels (crossbeam)**
- Automatic backpressure
- Well-tested library
- Good performance

#### Decision 4: Why Seeded RNG?

**Alternatives considered:**
- True randomness: not reproducible
- Pseudo-random without seed: hard to debug

**Chosen: ChaCha8 with seed**
- Deterministic
- Fast
- Good statistical properties
- Cryptographically secure (bonus)

### 11.3 Common Pitfalls and Solutions

#### Pitfall 1: Forgetting to Drop Channels

**Problem:**
```rust
let (tx, rx) = bounded(10);

std::thread::spawn(move || {
    for i in 0..100 {
        tx.send(i).unwrap();
    }
    // tx still alive!
});

// This hangs forever
for item in rx {
    println!("{}", item);
}
```

**Solution:**
```rust
std::thread::spawn(move || {
    for i in 0..100 {
        tx.send(i).unwrap();
    }
    drop(tx);  // Explicitly drop
});
```

#### Pitfall 2: Holding Mutex Too Long

**Problem:**
```rust
let data = shared.lock().unwrap();
expensive_computation(&data);  // Lock held entire time!
```

**Solution:**
```rust
let copy = {
    let data = shared.lock().unwrap();
    data.clone()
}; // Lock dropped here

expensive_computation(&copy);
```

#### Pitfall 3: Integer Overflow in Binary Format

**Problem:**
```rust
let count = symbols.len() as u8;  // Oops! 256 → 0
```

**Solution:**
```rust
let count = symbols.len() as u16;  // Can hold 0-65535
```

#### Pitfall 4: Off-by-One in Bit Indexing

**Problem:**
```rust
// MSB is bit 0 or bit 7?
let bit = (byte >> bit_index) & 1;
```

**Solution:**
```rust
// Document clearly: bit 0 = MSB
let bit = (byte >> (7 - bit_index)) & 1;
```

### 11.4 Extension Ideas

**Idea 1: Multiple Compression Algorithms**
- Add LZ77 for comparison
- Let user choose via CLI flag

**Idea 2: Real Network Support**
- Replace simulator with actual UDP sockets
- Test over real networks

**Idea 3: Visualization**
- Add web UI showing packets in flight
- Visualize Huffman tree
- Graph compression ratio over time

**Idea 4: Advanced Reassembly**
- Implement ARQ (Automatic Repeat Request)
- Add forward error correction
- Support packet retransmission

**Idea 5: Benchmarking Suite**
- Compare against gzip, bzip2, etc.
- Measure compression speed
- Test with various data types

### 11.5 Learning Roadmap

**Week 1: Fundamentals**
- Study information theory
- Implement BitWriter/BitReader
- Write comprehensive tests

**Week 2: Huffman Coding**
- Implement standard Huffman
- Convert to canonical form
- Test with various inputs

**Week 3: Framing & Packetization**
- Design binary format
- Implement serialization
- Add CRC validation

**Week 4: Network Simulation**
- Implement priority queue
- Add latency/jitter/loss
- Test determinism

**Week 5: Reassembly**
- Handle out-of-order packets
- Implement timeouts
- Ensure in-order delivery

**Week 6: Integration**
- Build complete pipeline
- Add threading
- Comprehensive testing

**Week 7: Polish**
- Add metrics
- Write documentation
- Performance tuning

### 11.6 Resources for Deep Learning

**Information Theory:**
- "Elements of Information Theory" by Cover & Thomas
- Online course: MIT 6.441 Information Theory

**Compression:**
- "Introduction to Data Compression" by Sayood
- RFC 1951 (DEFLATE specification)

**Networking:**
- "Computer Networks" by Tanenbaum
- RFC 793 (TCP specification)

**Systems Programming:**
- "The Rust Programming Language" (official book)
- "Programming Rust" by Blandy & Orendorff

**Concurrency:**
- "Rust Atomics and Locks" by m-ou-se
- "Concurrency in Practice" by Goetz

---

## Conclusion

This project demonstrates:
- **Compression** through canonical Huffman coding
- **Networking** through simulation of real-world impairments
- **Resilience** through reassembly and error handling
- **Systems design** through pipelined architecture

**Key takeaways:**
1. Information theory enables efficient encoding
2. Real networks are unreliable and require careful handling
3. Concurrent systems need bounded resources
4. Good error handling prevents cascading failures
5. Determinism aids debugging and testing

**You now have:**
- Complete understanding of the algorithms
- Detailed implementation knowledge
- Testing strategies
- Performance considerations
- Ability to reproduce and extend this project

**Next steps:**
1. Build it yourself from scratch
2. Experiment with different parameters
3. Add your own features
4. Compare with production systems

Remember: The goal is **learning**, not perfection. Every bug you encounter teaches you something valuable about systems programming.

Happy building! 🚀
