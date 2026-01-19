//! encoder-sim-core: Educational file transfer system with chunked Huffman compression
//!
//! This library provides the core components for a learning-focused system that:
//! - Compresses data using canonical Huffman coding
//! - Fragments compressed chunks into MTU-sized packets
//! - Simulates unreliable network conditions (latency, jitter, reordering, loss)
//! - Reassembles packets and validates integrity
//!
//! # Architecture
//!
//! The system is designed around clear module boundaries:
//! - `bitio`: Low-level bit reading/writing
//! - `huffman`: Canonical Huffman codec
//! - `framing`: Chunk frame serialization
//! - `packet`: Packet format and fragmentation
//! - `reassembly`: Packet reassembly with bounded memory
//! - `network`: Network simulator with seeded randomness
//! - `metrics`: Observable system behavior
//!
//! # Design Principles
//!
//! - **No panics**: All errors are structured and recoverable
//! - **Bounded memory**: Channels, buffers, and windows have fixed limits
//! - **Deterministic**: Seeded randomness makes runs reproducible
//! - **Observable**: Comprehensive metrics for understanding behavior

pub mod bitio;
pub mod error;
pub mod framing;
pub mod huffman;
pub mod metrics;
pub mod network;
pub mod packet;
pub mod reassembly;

// Re-export commonly used types
pub use error::{Error, Result};
