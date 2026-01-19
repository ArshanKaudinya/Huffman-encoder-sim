//! Bit-level I/O utilities for reading and writing individual bits.
//!
//! This module provides BitWriter and BitReader for serializing Huffman codes.
//! Both operate in MSB-first (most significant bit first) order, which is
//! standard for Huffman encoding.
//!
//! # Padding Rules
//! - BitWriter: pads incomplete bytes with trailing zeros
//! - BitReader: ignores padding bits at the end (caller must track exact bit count)
//!
//! # Example
//! ```
//! use encoder_sim_core::bitio::{BitWriter, BitReader};
//!
//! let mut writer = BitWriter::new();
//! writer.write_bits(0b101, 3).unwrap();  // Write 3 bits: 1, 0, 1
//! writer.write_bits(0b11, 2).unwrap();   // Write 2 bits: 1, 1
//! // Total: 10111 -> padded to 10111000
//!
//! let bytes = writer.finish();
//! let mut reader = BitReader::new(&bytes);
//! assert_eq!(reader.read_bits(3).unwrap(), 0b101);
//! assert_eq!(reader.read_bits(2).unwrap(), 0b11);
//! ```

use crate::error::{BitIoError, Result};

/// Writes bits MSB-first into a byte buffer.
///
/// Accumulates bits in a buffer and flushes complete bytes to the output.
/// When finished, pads the final partial byte with zeros.
///
/// # Invariants
/// - `bit_buffer` contains up to 7 bits (never a full byte)
/// - `bit_count` is always < 8
#[derive(Debug, Clone)]
pub struct BitWriter {
    /// Completed bytes
    bytes: Vec<u8>,
    /// Accumulator for the current partial byte (MSB-aligned)
    bit_buffer: u8,
    /// Number of bits in bit_buffer (0-7)
    bit_count: u8,
}

impl BitWriter {
    /// Create a new BitWriter with empty output.
    pub fn new() -> Self {
        Self {
            bytes: Vec::new(),
            bit_buffer: 0,
            bit_count: 0,
        }
    }

    /// Write up to 64 bits to the output.
    ///
    /// Bits are written MSB-first. For example, writing value=0b101 with count=3
    /// writes bits 1, 0, 1 in that order.
    ///
    /// # Arguments
    /// - `value`: the bits to write (only the lowest `count` bits are used)
    /// - `count`: number of bits to write (0-64)
    ///
    /// # Errors
    /// Returns `BitIoError::InvalidBitCount` if count > 64.
    pub fn write_bits(&mut self, value: u64, count: usize) -> Result<()> {
        if count > 64 {
            return Err(BitIoError::InvalidBitCount(count).into());
        }

        if count == 0 {
            return Ok(());
        }

        // Process bits from MSB to LSB of the value
        let mut remaining = count;
        let mut val = value;

        while remaining > 0 {
            // How many bits can we write to the current byte?
            let bits_to_write = remaining.min(8 - self.bit_count as usize);

            // Extract the top bits_to_write bits from val
            let shift = remaining - bits_to_write;
            let bits = ((val >> shift) & ((1 << bits_to_write) - 1)) as u8;

            // Add these bits to the buffer (shifted to align with current position)
            self.bit_buffer |= bits << (8 - self.bit_count as usize - bits_to_write);
            self.bit_count += bits_to_write as u8;

            // If we've filled a byte, flush it
            if self.bit_count == 8 {
                self.bytes.push(self.bit_buffer);
                self.bit_buffer = 0;
                self.bit_count = 0;
            }

            // Clear the bits we just wrote from val
            val &= (1 << shift) - 1;
            remaining -= bits_to_write;
        }

        Ok(())
    }

    /// Finish writing and return the output bytes.
    ///
    /// If there are any remaining bits in the buffer, they are padded with
    /// trailing zeros to complete the final byte.
    ///
    /// This consumes the writer.
    pub fn finish(mut self) -> Vec<u8> {
        // Flush any remaining bits (already padded with zeros)
        if self.bit_count > 0 {
            self.bytes.push(self.bit_buffer);
        }
        self.bytes
    }

    /// Return the number of complete bytes written so far.
    pub fn byte_len(&self) -> usize {
        self.bytes.len()
    }

    /// Return the total number of bits written (including partial byte).
    pub fn bit_len(&self) -> usize {
        self.bytes.len() * 8 + self.bit_count as usize
    }
}

impl Default for BitWriter {
    fn default() -> Self {
        Self::new()
    }
}

/// Reads bits MSB-first from a byte buffer.
///
/// Caller must track how many bits are valid; padding bits at the end
/// of the buffer are not distinguishable from data.
///
/// # Invariants
/// - `bit_position` never exceeds `data.len() * 8`
#[derive(Debug, Clone)]
pub struct BitReader<'a> {
    /// Source data
    data: &'a [u8],
    /// Current bit position (0 = MSB of first byte)
    bit_position: usize,
}

impl<'a> BitReader<'a> {
    /// Create a new BitReader for the given data.
    pub fn new(data: &'a [u8]) -> Self {
        Self {
            data,
            bit_position: 0,
        }
    }

    /// Read up to 64 bits from the input.
    ///
    /// Bits are read MSB-first. For example, reading 3 bits from byte 0b10110000
    /// returns 0b101.
    ///
    /// # Arguments
    /// - `count`: number of bits to read (0-64)
    ///
    /// # Errors
    /// - `BitIoError::InvalidBitCount` if count > 64
    /// - `BitIoError::UnexpectedEof` if not enough bits remain
    pub fn read_bits(&mut self, count: usize) -> Result<u64> {
        if count > 64 {
            return Err(BitIoError::InvalidBitCount(count).into());
        }

        if count == 0 {
            return Ok(0);
        }

        let available = self.bits_remaining();
        if count > available {
            return Err(BitIoError::UnexpectedEof.into());
        }

        let mut result = 0u64;
        let mut remaining = count;

        while remaining > 0 {
            let byte_idx = self.bit_position / 8;
            let bit_offset = self.bit_position % 8;

            // How many bits can we read from the current byte?
            let bits_in_byte = 8 - bit_offset;
            let bits_to_read = remaining.min(bits_in_byte);

            // Extract bits from current byte
            let byte = self.data[byte_idx];
            let mask = ((1u16 << bits_to_read) - 1) as u8;
            let bits = (byte >> (bits_in_byte - bits_to_read)) & mask;

            // Add to result
            result = (result << bits_to_read) | bits as u64;

            self.bit_position += bits_to_read;
            remaining -= bits_to_read;
        }

        Ok(result)
    }

    /// Read a single bit (0 or 1).
    pub fn read_bit(&mut self) -> Result<bool> {
        Ok(self.read_bits(1)? == 1)
    }

    /// Return the number of bits remaining in the buffer.
    pub fn bits_remaining(&self) -> usize {
        self.data.len() * 8 - self.bit_position
    }

    /// Return the current bit position.
    pub fn position(&self) -> usize {
        self.bit_position
    }

    /// Check if we're at the end of the buffer.
    pub fn is_empty(&self) -> bool {
        self.bit_position >= self.data.len() * 8
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_write_read_single_byte() {
        let mut writer = BitWriter::new();
        writer.write_bits(0b10110011, 8).unwrap();

        let bytes = writer.finish();
        assert_eq!(bytes, vec![0b10110011]);

        let mut reader = BitReader::new(&bytes);
        assert_eq!(reader.read_bits(8).unwrap(), 0b10110011);
    }

    #[test]
    fn test_write_read_partial_bits() {
        let mut writer = BitWriter::new();
        writer.write_bits(0b101, 3).unwrap();
        writer.write_bits(0b11, 2).unwrap();
        writer.write_bits(0b000, 3).unwrap();
        // Total: 10111000 (padded to byte boundary)

        let bytes = writer.finish();
        assert_eq!(bytes, vec![0b10111000]);

        let mut reader = BitReader::new(&bytes);
        assert_eq!(reader.read_bits(3).unwrap(), 0b101);
        assert_eq!(reader.read_bits(2).unwrap(), 0b11);
        assert_eq!(reader.read_bits(3).unwrap(), 0b000);
    }

    #[test]
    fn test_padding() {
        let mut writer = BitWriter::new();
        writer.write_bits(0b1, 1).unwrap();
        // Should be padded to 10000000

        let bytes = writer.finish();
        assert_eq!(bytes, vec![0b10000000]);
    }

    #[test]
    fn test_multi_byte() {
        let mut writer = BitWriter::new();
        // Write 16 bits across two bytes
        writer.write_bits(0b1010101111110000, 16).unwrap();

        let bytes = writer.finish();
        assert_eq!(bytes, vec![0b10101011, 0b11110000]);

        let mut reader = BitReader::new(&bytes);
        assert_eq!(reader.read_bits(16).unwrap(), 0b1010101111110000);
    }

    #[test]
    fn test_read_past_end() {
        let data = vec![0b10101010];
        let mut reader = BitReader::new(&data);

        assert_eq!(reader.read_bits(8).unwrap(), 0b10101010);
        assert!(reader.read_bits(1).is_err());
    }

    #[test]
    fn test_zero_bits() {
        let mut writer = BitWriter::new();
        writer.write_bits(0xFF, 0).unwrap();
        let bytes = writer.finish();
        assert_eq!(bytes.len(), 0);

        let mut reader = BitReader::new(&[0xFF]);
        assert_eq!(reader.read_bits(0).unwrap(), 0);
    }

    #[test]
    fn test_64_bit_values() {
        let mut writer = BitWriter::new();
        let val = 0x123456789ABCDEF0u64;
        writer.write_bits(val, 64).unwrap();

        let bytes = writer.finish();
        let mut reader = BitReader::new(&bytes);
        assert_eq!(reader.read_bits(64).unwrap(), val);
    }

    #[test]
    fn test_bit_by_bit() {
        let mut writer = BitWriter::new();
        for &bit in &[1u64, 0, 1, 1, 0, 0, 1, 0] {
            writer.write_bits(bit, 1).unwrap();
        }

        let bytes = writer.finish();
        assert_eq!(bytes, vec![0b10110010]);

        let mut reader = BitReader::new(&bytes);
        let expected = [true, false, true, true, false, false, true, false];
        for &exp in &expected {
            assert_eq!(reader.read_bit().unwrap(), exp);
        }
    }

    #[test]
    fn test_bits_remaining() {
        let data = vec![0xFF, 0xFF];
        let mut reader = BitReader::new(&data);

        assert_eq!(reader.bits_remaining(), 16);
        reader.read_bits(5).unwrap();
        assert_eq!(reader.bits_remaining(), 11);
        reader.read_bits(11).unwrap();
        assert_eq!(reader.bits_remaining(), 0);
        assert!(reader.is_empty());
    }
}
