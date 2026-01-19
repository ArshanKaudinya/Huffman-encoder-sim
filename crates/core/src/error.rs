//! Error types for the encoder-sim system.
//!
//! All operations return structured errors rather than panicking.
//! This enables graceful shutdown and clear error reporting.

use thiserror::Error;

/// Top-level error type for all operations in the system.
///
/// Each variant corresponds to a specific failure domain:
/// - Bit I/O: reading/writing bits from/to byte buffers
/// - Huffman: codec construction or encode/decode failures
/// - Framing: chunk frame serialization/parsing
/// - Packet: packet validation or fragmentation issues
/// - Reassembly: missing packets, timeouts, or out-of-order issues
/// - CRC: data corruption detected
/// - I/O: file system operations
#[derive(Debug, Error)]
pub enum Error {
    /// Bit I/O operation failed (e.g., reading past end of buffer)
    #[error("bit I/O error: {0}")]
    BitIo(#[from] BitIoError),

    /// Huffman codec error (e.g., invalid code, decode failure)
    #[error("huffman codec error: {0}")]
    Huffman(#[from] HuffmanError),

    /// Chunk frame error (e.g., invalid header, length mismatch)
    #[error("framing error: {0}")]
    Framing(#[from] FramingError),

    /// Packet validation or fragmentation error
    #[error("packet error: {0}")]
    Packet(#[from] PacketError),

    /// Reassembly error (e.g., timeout waiting for packets)
    #[error("reassembly error: {0}")]
    Reassembly(#[from] ReassemblyError),

    /// CRC validation failed, indicating data corruption
    #[error("CRC mismatch: expected {expected:#010x}, got {actual:#010x}")]
    Crc { expected: u32, actual: u32 },

    /// File I/O error
    #[error("I/O error: {0}")]
    Io(#[from] std::io::Error),

    /// Configuration error
    #[error("configuration error: {0}")]
    Config(String),

    /// Channel communication error
    #[error("channel error: {0}")]
    Channel(String),
}

/// Bit-level I/O errors.
#[derive(Debug, Error)]
pub enum BitIoError {
    /// Attempted to read past the end of the buffer
    #[error("unexpected end of bit stream")]
    UnexpectedEof,

    /// Attempted to read more bits than available
    #[error("insufficient bits: requested {requested}, available {available}")]
    InsufficientBits { requested: usize, available: usize },

    /// Invalid bit count (e.g., requesting 0 bits or more than 64 bits)
    #[error("invalid bit count: {0}")]
    InvalidBitCount(usize),
}

/// Huffman codec errors.
#[derive(Debug, Error)]
pub enum HuffmanError {
    /// No symbols with non-zero frequency (cannot build codebook)
    #[error("empty frequency table: cannot build codebook")]
    EmptyFrequencyTable,

    /// Decoded length doesn't match expected length
    #[error("decoded length mismatch: expected {expected}, got {actual}")]
    LengthMismatch { expected: usize, actual: usize },

    /// Invalid Huffman code encountered during decoding
    #[error("invalid huffman code at bit position {position}")]
    InvalidCode { position: usize },

    /// Code length exceeds maximum (255 bits)
    #[error("code length {length} exceeds maximum 255")]
    CodeLengthTooLong { length: usize },
}

/// Chunk framing errors.
#[derive(Debug, Error)]
pub enum FramingError {
    /// Invalid magic number in header
    #[error("invalid magic number: expected {expected:?}, got {actual:?}")]
    InvalidMagic { expected: [u8; 4], actual: [u8; 4] },

    /// Frame is too short to contain a valid header
    #[error("frame too short: need at least {required} bytes, got {actual}")]
    FrameTooShort { required: usize, actual: usize },

    /// Metadata length doesn't match header specification
    #[error("metadata length mismatch: header says {expected}, got {actual}")]
    MetadataLengthMismatch { expected: usize, actual: usize },

    /// Payload length doesn't match header specification
    #[error("payload length mismatch: header says {expected}, got {actual}")]
    PayloadLengthMismatch { expected: usize, actual: usize },
}

/// Packet errors.
#[derive(Debug, Error)]
pub enum PacketError {
    /// Invalid magic number in packet header
    #[error("invalid packet magic: expected {expected:?}, got {actual:?}")]
    InvalidMagic { expected: [u8; 2], actual: [u8; 2] },

    /// Packet is too short to contain a valid header
    #[error("packet too short: need at least {required} bytes, got {actual}")]
    PacketTooShort { required: usize, actual: usize },

    /// Packet ID is out of bounds (>= total_packets)
    #[error("packet_id {packet_id} >= total_packets {total_packets}")]
    InvalidPacketId {
        packet_id: u32,
        total_packets: u32,
    },

    /// Fragment length doesn't match actual data
    #[error("fragment length mismatch: header says {expected}, got {actual}")]
    FragmentLengthMismatch { expected: usize, actual: usize },

    /// Packet exceeds MTU
    #[error("packet size {size} exceeds MTU {mtu}")]
    ExceedsMtu { size: usize, mtu: usize },
}

/// Reassembly errors.
#[derive(Debug, Error)]
pub enum ReassemblyError {
    /// Timeout waiting for missing packets
    #[error("chunk {chunk_id} timed out after {timeout_ms}ms (missing {missing} packets)")]
    Timeout {
        chunk_id: u64,
        timeout_ms: u64,
        missing: usize,
    },

    /// Reassembly window is full (too many in-flight chunks)
    #[error("reassembly window full: max {max} chunks in flight")]
    WindowFull { max: usize },

    /// Duplicate packet received
    #[error("duplicate packet: chunk {chunk_id}, packet {packet_id}")]
    DuplicatePacket { chunk_id: u64, packet_id: u32 },

    /// Total packet count mismatch (different packets claim different totals)
    #[error("total packet count mismatch for chunk {chunk_id}: expected {expected}, got {actual}")]
    TotalPacketMismatch {
        chunk_id: u64,
        expected: u32,
        actual: u32,
    },
}

/// Type alias for Result with our Error type
pub type Result<T> = std::result::Result<T, Error>;
