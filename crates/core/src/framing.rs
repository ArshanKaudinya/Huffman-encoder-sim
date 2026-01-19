//! Chunk frame serialization and parsing.
//!
//! A chunk frame packages compressed data with metadata for transmission:
//! - Header: identifies the chunk, specifies lengths, includes CRC
//! - Codec metadata: Huffman codebook (symbol lengths)
//! - Payload: compressed bits
//!
//! # Frame Format
//!
//! ```text
//! +------------------+
//! | Magic (4 bytes)  |  0x48 0x46 0x46 0x54 ("HFFT")
//! +------------------+
//! | chunk_id (8)     |  u64 little-endian
//! +------------------+
//! | raw_len (4)      |  u32 original uncompressed size
//! +------------------+
//! | codec_meta_len(2)|  u16 length of huffman codebook
//! +------------------+
//! | payload_len (4)  |  u32 compressed bits length
//! +------------------+
//! | crc32 (4)        |  u32 checksum of everything
//! +------------------+
//! | codec_metadata   |  Huffman codebook (codec_meta_len bytes)
//! | (variable)       |
//! +------------------+
//! | payload          |  Compressed bits (payload_len bytes)
//! | (variable)       |
//! +------------------+
//! ```
//!
//! # CRC Coverage
//!
//! The CRC32 covers:
//! - chunk_id, raw_len, codec_meta_len, payload_len
//! - codec_metadata
//! - payload
//!
//! This detects corruption in headers or data.

use crate::error::{FramingError, Result};
use crate::huffman::Codebook;

/// Magic number for chunk frames: "HFFT" (Huffman File Transfer)
const MAGIC: [u8; 4] = [0x48, 0x46, 0x46, 0x54];

/// Size of the chunk frame header in bytes
const HEADER_SIZE: usize = 26;

/// A parsed chunk frame containing all metadata and payloads.
///
/// This is the result of deserializing a frame from bytes.
#[derive(Debug, Clone)]
pub struct ChunkFrame {
    /// Unique chunk identifier (sequential, starting from 0)
    pub chunk_id: u64,

    /// Original uncompressed data length
    pub raw_len: u32,

    /// Huffman codebook for decoding
    pub codebook: Codebook,

    /// Compressed payload (Huffman-encoded bits)
    pub payload: Vec<u8>,

    /// CRC32 checksum (for validation)
    pub crc32: u32,
}

/// Serialize a chunk frame into bytes for transmission.
///
/// # Arguments
/// - `chunk_id`: unique identifier for this chunk
/// - `raw_data`: original uncompressed data (used to compute raw_len)
/// - `compressed_payload`: Huffman-encoded bits
/// - `codebook`: the Huffman codebook used for encoding
///
/// # Returns
/// Complete serialized frame ready for packetization.
pub fn serialize_chunk_frame(
    chunk_id: u64,
    raw_data: &[u8],
    compressed_payload: &[u8],
    codebook: &Codebook,
) -> Vec<u8> {
    let codec_metadata = codebook.serialize_metadata();

    let raw_len = raw_data.len() as u32;
    let codec_meta_len = codec_metadata.len() as u16;
    let payload_len = compressed_payload.len() as u32;

    // Compute CRC over the data we're protecting
    let crc32 = compute_crc(
        chunk_id,
        raw_len,
        codec_meta_len,
        payload_len,
        &codec_metadata,
        compressed_payload,
    );

    // Allocate buffer for entire frame
    let total_size = HEADER_SIZE + codec_metadata.len() + compressed_payload.len();
    let mut frame = Vec::with_capacity(total_size);

    // Write header
    frame.extend_from_slice(&MAGIC);
    frame.extend_from_slice(&chunk_id.to_le_bytes());
    frame.extend_from_slice(&raw_len.to_le_bytes());
    frame.extend_from_slice(&codec_meta_len.to_le_bytes());
    frame.extend_from_slice(&payload_len.to_le_bytes());
    frame.extend_from_slice(&crc32.to_le_bytes());

    // Write metadata and payload
    frame.extend_from_slice(&codec_metadata);
    frame.extend_from_slice(compressed_payload);

    frame
}

/// Parse a chunk frame from bytes.
///
/// # Errors
/// - `FramingError::InvalidMagic` if magic number doesn't match
/// - `FramingError::FrameTooShort` if buffer is too small
/// - `Error::Crc` if CRC validation fails
/// - Propagates Huffman codebook deserialization errors
pub fn parse_chunk_frame(bytes: &[u8]) -> Result<ChunkFrame> {
    // Validate minimum size
    if bytes.len() < HEADER_SIZE {
        return Err(FramingError::FrameTooShort {
            required: HEADER_SIZE,
            actual: bytes.len(),
        }
        .into());
    }

    // Parse header
    let magic: [u8; 4] = bytes[0..4].try_into().unwrap();
    if magic != MAGIC {
        return Err(FramingError::InvalidMagic {
            expected: MAGIC,
            actual: magic,
        }
        .into());
    }

    let chunk_id = u64::from_le_bytes(bytes[4..12].try_into().unwrap());
    let raw_len = u32::from_le_bytes(bytes[12..16].try_into().unwrap());
    let codec_meta_len = u16::from_le_bytes(bytes[16..18].try_into().unwrap());
    let payload_len = u32::from_le_bytes(bytes[18..22].try_into().unwrap());
    let crc32 = u32::from_le_bytes(bytes[22..26].try_into().unwrap());

    // Validate total frame size
    let expected_size = HEADER_SIZE + codec_meta_len as usize + payload_len as usize;
    if bytes.len() != expected_size {
        return Err(FramingError::FrameTooShort {
            required: expected_size,
            actual: bytes.len(),
        }
        .into());
    }

    // Extract metadata and payload
    let meta_start = HEADER_SIZE;
    let meta_end = meta_start + codec_meta_len as usize;
    let payload_start = meta_end;
    let payload_end = payload_start + payload_len as usize;

    let codec_metadata = &bytes[meta_start..meta_end];
    let payload = &bytes[payload_start..payload_end];

    // Verify CRC
    let computed_crc = compute_crc(
        chunk_id,
        raw_len,
        codec_meta_len,
        payload_len,
        codec_metadata,
        payload,
    );

    if computed_crc != crc32 {
        return Err(crate::error::Error::Crc {
            expected: crc32,
            actual: computed_crc,
        });
    }

    // Deserialize codebook
    let codebook = Codebook::deserialize_metadata(codec_metadata)?;

    Ok(ChunkFrame {
        chunk_id,
        raw_len,
        codebook,
        payload: payload.to_vec(),
        crc32,
    })
}

/// Compute CRC32 over the protected fields.
///
/// This function defines what data is covered by the integrity check.
fn compute_crc(
    chunk_id: u64,
    raw_len: u32,
    codec_meta_len: u16,
    payload_len: u32,
    codec_metadata: &[u8],
    payload: &[u8],
) -> u32 {
    let mut hasher = crc32fast::Hasher::new();

    // Hash header fields
    hasher.update(&chunk_id.to_le_bytes());
    hasher.update(&raw_len.to_le_bytes());
    hasher.update(&codec_meta_len.to_le_bytes());
    hasher.update(&payload_len.to_le_bytes());

    // Hash metadata and payload
    hasher.update(codec_metadata);
    hasher.update(payload);

    hasher.finalize()
}

/// Compress raw data and build a chunk frame.
///
/// This is a convenience function that combines Huffman encoding with framing.
///
/// # Arguments
/// - `chunk_id`: unique chunk identifier
/// - `raw_data`: uncompressed input data
///
/// # Returns
/// Serialized chunk frame ready for packetization.
pub fn compress_and_frame(chunk_id: u64, raw_data: &[u8]) -> Result<Vec<u8>> {
    // Build frequency table
    let mut freqs = [0u64; 256];
    for &byte in raw_data {
        freqs[byte as usize] += 1;
    }

    // Build codebook and encode
    let codebook = Codebook::from_frequencies(&freqs)?;
    let compressed = codebook.encode(raw_data)?;

    // Serialize frame
    Ok(serialize_chunk_frame(
        chunk_id,
        raw_data,
        &compressed,
        &codebook,
    ))
}

/// Decompress a parsed chunk frame.
///
/// # Returns
/// The original uncompressed data.
pub fn decompress_frame(frame: &ChunkFrame) -> Result<Vec<u8>> {
    frame
        .codebook
        .decode(&frame.payload, frame.raw_len as usize)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_serialize_parse_round_trip() {
        let chunk_id = 42;
        let raw_data = b"hello world! this is a test.";

        // Compress and frame
        let frame_bytes = compress_and_frame(chunk_id, raw_data).unwrap();

        // Parse
        let frame = parse_chunk_frame(&frame_bytes).unwrap();

        // Validate header fields
        assert_eq!(frame.chunk_id, chunk_id);
        assert_eq!(frame.raw_len, raw_data.len() as u32);

        // Decompress and verify
        let decompressed = decompress_frame(&frame).unwrap();
        assert_eq!(decompressed, raw_data);
    }

    #[test]
    fn test_invalid_magic() {
        let mut frame_bytes = vec![0xFF, 0xFF, 0xFF, 0xFF]; // Wrong magic
        frame_bytes.extend_from_slice(&[0u8; 22]); // Rest of header

        let result = parse_chunk_frame(&frame_bytes);
        assert!(matches!(
            result,
            Err(crate::error::Error::Framing(FramingError::InvalidMagic { .. }))
        ));
    }

    #[test]
    fn test_frame_too_short() {
        let frame_bytes = vec![0u8; 10]; // Less than HEADER_SIZE
        let result = parse_chunk_frame(&frame_bytes);
        assert!(matches!(
            result,
            Err(crate::error::Error::Framing(FramingError::FrameTooShort { .. }))
        ));
    }

    #[test]
    fn test_crc_mismatch() {
        let chunk_id = 1;
        let raw_data = b"test data";

        let mut frame_bytes = compress_and_frame(chunk_id, raw_data).unwrap();

        // Corrupt a byte in the payload (near the end)
        let len = frame_bytes.len();
        frame_bytes[len - 1] ^= 0x01;

        let result = parse_chunk_frame(&frame_bytes);
        assert!(matches!(result, Err(crate::error::Error::Crc { .. })));
    }

    #[test]
    fn test_empty_data() {
        // Empty data should fail (no symbols to encode)
        let result = compress_and_frame(0, b"");
        assert!(result.is_err());
    }

    #[test]
    fn test_single_byte() {
        let chunk_id = 0;
        let raw_data = b"A";

        let frame_bytes = compress_and_frame(chunk_id, raw_data).unwrap();
        let frame = parse_chunk_frame(&frame_bytes).unwrap();
        let decompressed = decompress_frame(&frame).unwrap();

        assert_eq!(decompressed, raw_data);
    }

    #[test]
    fn test_large_data() {
        let chunk_id = 99;
        let raw_data = vec![b'X'; 65536]; // 64 KiB of same byte

        let frame_bytes = compress_and_frame(chunk_id, &raw_data).unwrap();

        // Should compress well (single symbol)
        assert!(frame_bytes.len() < raw_data.len() / 2);

        let frame = parse_chunk_frame(&frame_bytes).unwrap();
        let decompressed = decompress_frame(&frame).unwrap();

        assert_eq!(decompressed, raw_data);
    }
}
