//! Packet format and fragmentation.
//!
//! Large chunk frames are fragmented into MTU-sized packets for transmission.
//! Each packet contains:
//! - Header identifying which chunk and which fragment
//! - A slice of the chunk frame data
//!
//! # Packet Format
//!
//! ```text
//! +-------------------+
//! | Magic (2 bytes)   |  0x50 0x4B ("PK")
//! +-------------------+
//! | chunk_id (8)      |  u64 which chunk this belongs to
//! +-------------------+
//! | packet_id (4)     |  u32 fragment index (0-based)
//! +-------------------+
//! | total_packets (4) |  u32 total fragments for this chunk
//! +-------------------+
//! | fragment_len (2)  |  u16 bytes in fragment
//! +-------------------+
//! | fragment          |  fragment_len bytes of chunk frame
//! | (variable)        |
//! +-------------------+
//! ```
//!
//! # MTU Constraint
//!
//! Total packet size = HEADER_SIZE + fragment_len <= MTU
//!
//! # Reassembly
//!
//! The receiver collects all packets for a chunk_id, validates packet_ids are
//! in range [0, total_packets), and concatenates fragments in order.

use crate::error::{PacketError, Result};

/// Magic number for packets: "PK"
const MAGIC: [u8; 2] = [0x50, 0x4B];

/// Size of packet header in bytes
pub const HEADER_SIZE: usize = 20;

/// A network packet containing a fragment of a chunk frame.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Packet {
    /// Which chunk this packet belongs to
    pub chunk_id: u64,

    /// Packet index within the chunk (0-based)
    pub packet_id: u32,

    /// Total number of packets for this chunk
    pub total_packets: u32,

    /// Fragment of the chunk frame
    pub fragment: Vec<u8>,
}

impl Packet {
    /// Create a new packet.
    ///
    /// This is primarily used by the packetization logic.
    pub fn new(chunk_id: u64, packet_id: u32, total_packets: u32, fragment: Vec<u8>) -> Self {
        Self {
            chunk_id,
            packet_id,
            total_packets,
            fragment,
        }
    }

    /// Serialize this packet into bytes for transmission.
    ///
    /// # Returns
    /// Wire format bytes ready to send over the network.
    pub fn serialize(&self) -> Vec<u8> {
        let fragment_len = self.fragment.len() as u16;
        let total_size = HEADER_SIZE + self.fragment.len();

        let mut bytes = Vec::with_capacity(total_size);

        // Write header
        bytes.extend_from_slice(&MAGIC);
        bytes.extend_from_slice(&self.chunk_id.to_le_bytes());
        bytes.extend_from_slice(&self.packet_id.to_le_bytes());
        bytes.extend_from_slice(&self.total_packets.to_le_bytes());
        bytes.extend_from_slice(&fragment_len.to_le_bytes());

        // Write fragment
        bytes.extend_from_slice(&self.fragment);

        bytes
    }

    /// Deserialize a packet from bytes.
    ///
    /// # Errors
    /// - `PacketError::InvalidMagic` if magic doesn't match
    /// - `PacketError::PacketTooShort` if buffer is too small
    /// - `PacketError::InvalidPacketId` if packet_id >= total_packets
    /// - `PacketError::FragmentLengthMismatch` if fragment length doesn't match
    pub fn deserialize(bytes: &[u8]) -> Result<Self> {
        // Validate minimum size
        if bytes.len() < HEADER_SIZE {
            return Err(PacketError::PacketTooShort {
                required: HEADER_SIZE,
                actual: bytes.len(),
            }
            .into());
        }

        // Parse header
        let magic: [u8; 2] = bytes[0..2].try_into().unwrap();
        if magic != MAGIC {
            return Err(PacketError::InvalidMagic {
                expected: MAGIC,
                actual: magic,
            }
            .into());
        }

        let chunk_id = u64::from_le_bytes(bytes[2..10].try_into().unwrap());
        let packet_id = u32::from_le_bytes(bytes[10..14].try_into().unwrap());
        let total_packets = u32::from_le_bytes(bytes[14..18].try_into().unwrap());
        let fragment_len = u16::from_le_bytes(bytes[18..20].try_into().unwrap());

        // Validate packet_id
        if packet_id >= total_packets {
            return Err(PacketError::InvalidPacketId {
                packet_id,
                total_packets,
            }
            .into());
        }

        // Validate total size
        let expected_size = HEADER_SIZE + fragment_len as usize;
        if bytes.len() != expected_size {
            return Err(PacketError::FragmentLengthMismatch {
                expected: fragment_len as usize,
                actual: bytes.len() - HEADER_SIZE,
            }
            .into());
        }

        // Extract fragment
        let fragment = bytes[HEADER_SIZE..].to_vec();

        Ok(Self {
            chunk_id,
            packet_id,
            total_packets,
            fragment,
        })
    }

    /// Get the total size of this packet when serialized.
    pub fn size(&self) -> usize {
        HEADER_SIZE + self.fragment.len()
    }

    /// Validate that this packet's size doesn't exceed MTU.
    pub fn validate_mtu(&self, mtu: usize) -> Result<()> {
        let size = self.size();
        if size > mtu {
            return Err(PacketError::ExceedsMtu { size, mtu }.into());
        }
        Ok(())
    }
}

/// Fragment a chunk frame into MTU-sized packets.
///
/// # Arguments
/// - `chunk_id`: unique identifier for the chunk
/// - `chunk_frame`: serialized chunk frame bytes
/// - `mtu`: maximum transmission unit (packet size limit)
///
/// # Returns
/// Vector of packets, each with size <= MTU.
///
/// # Algorithm
/// 1. Calculate maximum fragment size per packet: `mtu - HEADER_SIZE`
/// 2. Split chunk frame into fragments of this size
/// 3. Create packet for each fragment with appropriate metadata
///
/// # Errors
/// Returns error if MTU is too small to fit even a header.
pub fn packetize(chunk_id: u64, chunk_frame: &[u8], mtu: usize) -> Result<Vec<Packet>> {
    // Validate MTU is reasonable
    if mtu < HEADER_SIZE + 1 {
        return Err(PacketError::ExceedsMtu {
            size: HEADER_SIZE + 1,
            mtu,
        }
        .into());
    }

    // Maximum bytes we can fit in each packet's fragment
    let max_fragment_size = mtu - HEADER_SIZE;

    // Calculate total packets needed
    let total_packets = (chunk_frame.len() + max_fragment_size - 1) / max_fragment_size;

    let mut packets = Vec::with_capacity(total_packets);

    for packet_id in 0..total_packets {
        let start = packet_id * max_fragment_size;
        let end = (start + max_fragment_size).min(chunk_frame.len());
        let fragment = chunk_frame[start..end].to_vec();

        let packet = Packet::new(chunk_id, packet_id as u32, total_packets as u32, fragment);

        // Validate MTU (should always pass, but defensive)
        packet.validate_mtu(mtu)?;

        packets.push(packet);
    }

    Ok(packets)
}

/// Reassemble a chunk frame from packets.
///
/// # Arguments
/// - `packets`: collection of packets for a single chunk (must all have same chunk_id)
///
/// # Returns
/// Reconstructed chunk frame bytes.
///
/// # Errors
/// - `PacketError::InvalidPacketId` if packet_ids are inconsistent
/// - Error if total_packets values don't match across packets
///
/// # Preconditions
/// Caller must ensure:
/// - All packets have the same chunk_id
/// - All packets are present (no gaps)
/// - Packets may be in any order (this function sorts them)
pub fn reassemble(packets: &mut [Packet]) -> Result<Vec<u8>> {
    if packets.is_empty() {
        return Ok(Vec::new());
    }

    // Validate all packets have same chunk_id and total_packets
    let chunk_id = packets[0].chunk_id;
    let total_packets = packets[0].total_packets;

    for packet in packets.iter() {
        if packet.chunk_id != chunk_id {
            // This is a logic error - caller should have filtered by chunk_id
            return Err(PacketError::InvalidPacketId {
                packet_id: packet.packet_id,
                total_packets,
            }
            .into());
        }
    }

    // Sort packets by packet_id
    packets.sort_by_key(|p| p.packet_id);

    // Validate we have all packets in sequence
    if packets.len() != total_packets as usize {
        return Err(PacketError::InvalidPacketId {
            packet_id: packets.len() as u32,
            total_packets,
        }
        .into());
    }

    for (i, packet) in packets.iter().enumerate() {
        if packet.packet_id != i as u32 {
            return Err(PacketError::InvalidPacketId {
                packet_id: packet.packet_id,
                total_packets,
            }
            .into());
        }
    }

    // Concatenate fragments
    let total_size: usize = packets.iter().map(|p| p.fragment.len()).sum();
    let mut chunk_frame = Vec::with_capacity(total_size);

    for packet in packets {
        chunk_frame.extend_from_slice(&packet.fragment);
    }

    Ok(chunk_frame)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_packet_serialize_deserialize() {
        let packet = Packet::new(42, 3, 10, vec![0xAA, 0xBB, 0xCC]);

        let bytes = packet.serialize();
        let deserialized = Packet::deserialize(&bytes).unwrap();

        assert_eq!(deserialized, packet);
    }

    #[test]
    fn test_invalid_magic() {
        let mut bytes = vec![0xFF, 0xFF]; // Wrong magic
        bytes.extend_from_slice(&[0u8; 18]); // Rest of header
        bytes.push(0); // Fragment

        let result = Packet::deserialize(&bytes);
        assert!(matches!(
            result,
            Err(crate::error::Error::Packet(PacketError::InvalidMagic { .. }))
        ));
    }

    #[test]
    fn test_packet_too_short() {
        let bytes = vec![0x50, 0x4B, 0, 0, 0]; // Only 5 bytes
        let result = Packet::deserialize(&bytes);
        assert!(matches!(
            result,
            Err(crate::error::Error::Packet(PacketError::PacketTooShort { .. }))
        ));
    }

    #[test]
    fn test_invalid_packet_id() {
        let packet = Packet::new(1, 10, 5, vec![0]); // packet_id >= total_packets
        let bytes = packet.serialize();

        let result = Packet::deserialize(&bytes);
        assert!(matches!(
            result,
            Err(crate::error::Error::Packet(PacketError::InvalidPacketId { .. }))
        ));
    }

    #[test]
    fn test_packetize_single_packet() {
        let chunk_frame = vec![0xAA; 100];
        let mtu = 200;

        let packets = packetize(42, &chunk_frame, mtu).unwrap();

        assert_eq!(packets.len(), 1);
        assert_eq!(packets[0].chunk_id, 42);
        assert_eq!(packets[0].packet_id, 0);
        assert_eq!(packets[0].total_packets, 1);
        assert_eq!(packets[0].fragment.len(), 100);
    }

    #[test]
    fn test_packetize_multiple_packets() {
        let chunk_frame = vec![0xBB; 300];
        let mtu = 120; // HEADER_SIZE=20, so max_fragment=100

        let packets = packetize(99, &chunk_frame, mtu).unwrap();

        // Should need 3 packets: 100 + 100 + 100
        assert_eq!(packets.len(), 3);

        for (i, packet) in packets.iter().enumerate() {
            assert_eq!(packet.chunk_id, 99);
            assert_eq!(packet.packet_id, i as u32);
            assert_eq!(packet.total_packets, 3);
            assert!(packet.size() <= mtu);
        }

        // First two should be full, last one exactly 100
        assert_eq!(packets[0].fragment.len(), 100);
        assert_eq!(packets[1].fragment.len(), 100);
        assert_eq!(packets[2].fragment.len(), 100);
    }

    #[test]
    fn test_packetize_exact_boundary() {
        let mtu = 120;
        let max_fragment = mtu - HEADER_SIZE; // 100
        let chunk_frame = vec![0xCC; max_fragment * 2]; // Exactly 2 packets

        let packets = packetize(1, &chunk_frame, mtu).unwrap();

        assert_eq!(packets.len(), 2);
        assert_eq!(packets[0].fragment.len(), max_fragment);
        assert_eq!(packets[1].fragment.len(), max_fragment);
    }

    #[test]
    fn test_packetize_reassemble_round_trip() {
        let chunk_frame = vec![0x42; 500];
        let mtu = 150;

        let mut packets = packetize(77, &chunk_frame, mtu).unwrap();
        let reassembled = reassemble(&mut packets).unwrap();

        assert_eq!(reassembled, chunk_frame);
    }

    #[test]
    fn test_reassemble_out_of_order() {
        let chunk_frame = vec![0x01, 0x02, 0x03, 0x04, 0x05, 0x06];
        let mtu = 23; // Small MTU to force multiple packets

        let mut packets = packetize(1, &chunk_frame, mtu).unwrap();

        // Shuffle packets
        packets.reverse();

        let reassembled = reassemble(&mut packets).unwrap();
        assert_eq!(reassembled, chunk_frame);
    }

    #[test]
    fn test_mtu_too_small() {
        let chunk_frame = vec![0; 10];
        let mtu = HEADER_SIZE; // No room for fragment

        let result = packetize(1, &chunk_frame, mtu);
        assert!(result.is_err());
    }

    #[test]
    fn test_empty_chunk_frame() {
        let chunk_frame = vec![];
        let mtu = 100;

        let packets = packetize(1, &chunk_frame, mtu).unwrap();
        assert_eq!(packets.len(), 0);
    }

    #[test]
    fn test_reassemble_missing_packet() {
        let mut packets = vec![
            Packet::new(1, 0, 3, vec![0]),
            Packet::new(1, 2, 3, vec![0]), // Missing packet 1
        ];

        let result = reassemble(&mut packets);
        assert!(result.is_err());
    }
}
