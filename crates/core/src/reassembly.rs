//! Packet reassembly with bounded memory.
//!
//! The reassembler maintains a sliding window of in-flight chunks, collecting
//! packets for each chunk until complete. This implements flow control and
//! prevents unbounded memory growth.
//!
//! # Design
//!
//! - **Bounded window**: Track at most `max_inflight_chunks` simultaneously
//! - **Per-chunk state**: For each chunk, track received packets
//! - **Completion detection**: When all packets received, emit complete chunk
//! - **Timeout handling**: Caller-driven timeout for detecting packet loss
//!
//! # Memory Bounds
//!
//! Maximum memory usage:
//! ```text
//! max_inflight_chunks * max_packets_per_chunk * max_packet_size
//! ```
//!
//! With typical values (64 chunks, 128 packets/chunk, 1200 bytes/packet):
//! ~10 MiB maximum buffering
//!
//! # Thread Safety
//!
//! This structure is NOT thread-safe. Caller must synchronize access or
//! use per-thread instances.

use crate::error::{ReassemblyError, Result};
use crate::packet::Packet;
use std::collections::HashMap;
use std::time::{Duration, Instant};

/// State for a single chunk being reassembled.
#[derive(Debug)]
struct ChunkState {
    /// Expected total number of packets
    total_packets: u32,

    /// Packets received so far (indexed by packet_id)
    packets: HashMap<u32, Packet>,

    /// When we first received a packet for this chunk (for timeout)
    first_seen: Instant,
}

impl ChunkState {
    /// Create new chunk state when first packet arrives.
    fn new(total_packets: u32) -> Self {
        Self {
            total_packets,
            packets: HashMap::new(),
            first_seen: Instant::now(),
        }
    }

    /// Insert a packet and return true if chunk is now complete.
    fn insert_packet(&mut self, packet: Packet) -> Result<bool> {
        let packet_id = packet.packet_id;

        // Validate total_packets matches
        if packet.total_packets != self.total_packets {
            return Err(ReassemblyError::TotalPacketMismatch {
                chunk_id: packet.chunk_id,
                expected: self.total_packets,
                actual: packet.total_packets,
            }
            .into());
        }

        // Check for duplicate
        if self.packets.contains_key(&packet_id) {
            return Err(ReassemblyError::DuplicatePacket {
                chunk_id: packet.chunk_id,
                packet_id,
            }
            .into());
        }

        self.packets.insert(packet_id, packet);

        // Check if complete
        Ok(self.packets.len() == self.total_packets as usize)
    }

    /// Extract all packets in sorted order (consumes the state).
    fn take_packets(self) -> Vec<Packet> {
        let mut packets: Vec<_> = self.packets.into_values().collect();
        packets.sort_by_key(|p| p.packet_id);
        packets
    }

    /// Check if this chunk has timed out.
    fn is_timed_out(&self, timeout: Duration) -> bool {
        self.first_seen.elapsed() >= timeout
    }

    /// Number of packets received so far.
    #[allow(dead_code)]
    fn received_count(&self) -> usize {
        self.packets.len()
    }

    /// Number of missing packets.
    fn missing_count(&self) -> usize {
        self.total_packets as usize - self.packets.len()
    }
}

/// Reassembler for collecting packets back into chunk frames.
///
/// Maintains bounded state for multiple in-flight chunks.
pub struct Reassembler {
    /// Maximum chunks we can track simultaneously
    max_inflight: usize,

    /// Timeout for incomplete chunks
    timeout: Duration,

    /// In-flight chunks indexed by chunk_id
    chunks: HashMap<u64, ChunkState>,

    /// Next chunk_id we expect to emit (for in-order delivery)
    next_expected_chunk: u64,

    /// Completed chunks waiting to be emitted in order
    completed_chunks: HashMap<u64, Vec<u8>>,
}

impl Reassembler {
    /// Create a new reassembler.
    ///
    /// # Arguments
    /// - `max_inflight`: maximum number of chunks to track simultaneously
    /// - `timeout_ms`: milliseconds to wait for missing packets before giving up
    pub fn new(max_inflight: usize, timeout_ms: u64) -> Self {
        Self {
            max_inflight,
            timeout: Duration::from_millis(timeout_ms),
            chunks: HashMap::new(),
            next_expected_chunk: 0,
            completed_chunks: HashMap::new(),
        }
    }

    /// Insert a packet and potentially return completed chunk frames.
    ///
    /// # Returns
    /// - `Ok(Some((chunk_id, frame_bytes)))` if a chunk was completed and is next in sequence
    /// - `Ok(None)` if packet was buffered but chunk not yet complete
    ///
    /// # Errors
    /// - `ReassemblyError::WindowFull` if too many chunks in flight
    /// - `ReassemblyError::DuplicatePacket` if packet already received
    /// - `ReassemblyError::TotalPacketMismatch` if total_packets doesn't match
    ///
    /// # Behavior
    ///
    /// Chunks are emitted in order. If chunk N+1 completes before chunk N,
    /// it's buffered until N completes.
    pub fn insert_packet(&mut self, packet: Packet) -> Result<Option<(u64, Vec<u8>)>> {
        let chunk_id = packet.chunk_id;

        // Check if this chunk is already completed and waiting
        if self.completed_chunks.contains_key(&chunk_id) {
            // Duplicate packet for already-completed chunk; ignore
            return Ok(None);
        }

        // Check if this chunk is before our window (already emitted)
        if chunk_id < self.next_expected_chunk {
            // Late arrival of packet for already-processed chunk; ignore
            return Ok(None);
        }

        // Get or create chunk state
        let is_new_chunk = !self.chunks.contains_key(&chunk_id);

        if is_new_chunk {
            // Check if we have room for a new chunk
            if self.chunks.len() >= self.max_inflight {
                return Err(ReassemblyError::WindowFull {
                    max: self.max_inflight,
                }
                .into());
            }

            self.chunks
                .insert(chunk_id, ChunkState::new(packet.total_packets));
        }

        // Insert packet into chunk
        let chunk_state = self.chunks.get_mut(&chunk_id).unwrap();
        let is_complete = chunk_state.insert_packet(packet)?;

        if is_complete {
            // Remove from in-flight and reassemble
            let chunk_state = self.chunks.remove(&chunk_id).unwrap();
            let mut packets = chunk_state.take_packets();
            let frame_bytes = crate::packet::reassemble(&mut packets)?;

            // Store in completed buffer
            self.completed_chunks.insert(chunk_id, frame_bytes);

            // Try to emit in-order chunks
            return Ok(self.try_emit_next());
        }

        Ok(None)
    }

    /// Try to emit the next in-order chunk if available.
    ///
    /// Returns Some((chunk_id, bytes)) if next chunk is ready.
    fn try_emit_next(&mut self) -> Option<(u64, Vec<u8>)> {
        if let Some(bytes) = self.completed_chunks.remove(&self.next_expected_chunk) {
            let chunk_id = self.next_expected_chunk;
            self.next_expected_chunk += 1;
            Some((chunk_id, bytes))
        } else {
            None
        }
    }

    /// Check for timed out chunks and return errors.
    ///
    /// Caller should call this periodically to detect packet loss.
    ///
    /// # Returns
    /// Vector of chunk_ids that have timed out.
    ///
    /// # Side Effects
    /// Timed-out chunks are removed from the reassembler state.
    pub fn check_timeouts(&mut self) -> Vec<ReassemblyError> {
        let mut errors = Vec::new();

        // Find timed-out chunks
        let timed_out: Vec<u64> = self
            .chunks
            .iter()
            .filter(|(_, state)| state.is_timed_out(self.timeout))
            .map(|(chunk_id, _)| *chunk_id)
            .collect();

        // Remove them and generate errors
        for chunk_id in timed_out {
            if let Some(state) = self.chunks.remove(&chunk_id) {
                errors.push(ReassemblyError::Timeout {
                    chunk_id,
                    timeout_ms: self.timeout.as_millis() as u64,
                    missing: state.missing_count(),
                });
            }
        }

        errors
    }

    /// Attempt to emit completed chunks that are now in order.
    ///
    /// Call this after handling timeouts to drain any chunks that
    /// became emittable.
    ///
    /// # Returns
    /// Vector of (chunk_id, frame_bytes) pairs in order.
    pub fn drain_completed(&mut self) -> Vec<(u64, Vec<u8>)> {
        let mut result = Vec::new();

        while let Some(chunk) = self.try_emit_next() {
            result.push(chunk);
        }

        result
    }

    /// Get statistics about current reassembly state.
    pub fn stats(&self) -> ReassemblerStats {
        ReassemblerStats {
            inflight_chunks: self.chunks.len(),
            completed_waiting: self.completed_chunks.len(),
            next_expected_chunk: self.next_expected_chunk,
        }
    }

    /// Check if reassembler is idle (no chunks in flight or waiting).
    pub fn is_idle(&self) -> bool {
        self.chunks.is_empty() && self.completed_chunks.is_empty()
    }
}

/// Statistics about reassembler state.
#[derive(Debug, Clone, Copy)]
pub struct ReassemblerStats {
    /// Number of chunks currently being reassembled
    pub inflight_chunks: usize,

    /// Number of completed chunks waiting for in-order emission
    pub completed_waiting: usize,

    /// Next chunk ID expected to be emitted
    pub next_expected_chunk: u64,
}

#[cfg(test)]
mod tests {
    use super::*;

    fn make_packet(chunk_id: u64, packet_id: u32, total: u32) -> Packet {
        Packet::new(chunk_id, packet_id, total, vec![packet_id as u8])
    }

    #[test]
    fn test_single_chunk_in_order() {
        let mut reassembler = Reassembler::new(10, 1000);

        // Send 3 packets for chunk 0
        assert!(reassembler.insert_packet(make_packet(0, 0, 3)).unwrap().is_none());
        assert!(reassembler.insert_packet(make_packet(0, 1, 3)).unwrap().is_none());

        // Last packet should complete the chunk
        let result = reassembler.insert_packet(make_packet(0, 2, 3)).unwrap();
        assert!(result.is_some());

        let (chunk_id, _bytes) = result.unwrap();
        assert_eq!(chunk_id, 0);
    }

    #[test]
    fn test_out_of_order_packets() {
        let mut reassembler = Reassembler::new(10, 1000);

        // Send packets out of order
        assert!(reassembler.insert_packet(make_packet(0, 2, 3)).unwrap().is_none());
        assert!(reassembler.insert_packet(make_packet(0, 0, 3)).unwrap().is_none());

        // Complete
        let result = reassembler.insert_packet(make_packet(0, 1, 3)).unwrap();
        assert!(result.is_some());
    }

    #[test]
    fn test_duplicate_packet() {
        let mut reassembler = Reassembler::new(10, 1000);

        reassembler.insert_packet(make_packet(0, 0, 3)).unwrap();

        // Duplicate should error
        let result = reassembler.insert_packet(make_packet(0, 0, 3));
        assert!(matches!(
            result,
            Err(crate::error::Error::Reassembly(ReassemblyError::DuplicatePacket { .. }))
        ));
    }

    #[test]
    fn test_window_full() {
        let mut reassembler = Reassembler::new(2, 1000);

        // Start 2 chunks
        reassembler.insert_packet(make_packet(0, 0, 2)).unwrap();
        reassembler.insert_packet(make_packet(1, 0, 2)).unwrap();

        // Third chunk should fail
        let result = reassembler.insert_packet(make_packet(2, 0, 2));
        assert!(matches!(
            result,
            Err(crate::error::Error::Reassembly(ReassemblyError::WindowFull { .. }))
        ));
    }

    #[test]
    fn test_in_order_delivery() {
        let mut reassembler = Reassembler::new(10, 1000);

        // Complete chunk 1 first (before chunk 0)
        let result = reassembler.insert_packet(make_packet(1, 0, 1)).unwrap();
        // Should not emit yet (waiting for chunk 0)
        assert!(result.is_none());

        // Try to insert duplicate packet for completed chunk 1
        let result = reassembler.insert_packet(make_packet(1, 0, 1)).unwrap();
        // Should be ignored (lenient duplicate handling for completed chunks)
        assert!(result.is_none());

        // Stats should show 1 completed waiting
        let stats = reassembler.stats();
        assert_eq!(stats.completed_waiting, 1);
        assert_eq!(stats.next_expected_chunk, 0);

        // Now complete chunk 0
        let result = reassembler.insert_packet(make_packet(0, 0, 1)).unwrap();

        // Should emit chunk 0
        assert!(result.is_some());
        let (chunk_id, _) = result.unwrap();
        assert_eq!(chunk_id, 0);

        // Now drain should emit chunk 1
        let drained = reassembler.drain_completed();
        assert_eq!(drained.len(), 1);
        assert_eq!(drained[0].0, 1);
    }

    #[test]
    fn test_timeout() {
        let mut reassembler = Reassembler::new(10, 50); // 50ms timeout

        // Start chunk 0 but don't complete
        reassembler.insert_packet(make_packet(0, 0, 3)).unwrap();

        // Initially no timeout
        assert_eq!(reassembler.check_timeouts().len(), 0);

        // Wait for timeout
        std::thread::sleep(Duration::from_millis(60));

        // Should timeout
        let errors = reassembler.check_timeouts();
        assert_eq!(errors.len(), 1);

        match &errors[0] {
            ReassemblyError::Timeout { chunk_id, missing, .. } => {
                assert_eq!(*chunk_id, 0);
                assert_eq!(*missing, 2); // Missing 2 of 3 packets
            }
            _ => panic!("Expected timeout error"),
        }

        // Chunk should be removed
        assert!(reassembler.is_idle());
    }

    #[test]
    fn test_total_packet_mismatch() {
        let mut reassembler = Reassembler::new(10, 1000);

        reassembler.insert_packet(make_packet(0, 0, 3)).unwrap();

        // Send packet with different total
        let bad_packet = Packet::new(0, 1, 5, vec![1]); // total=5 instead of 3
        let result = reassembler.insert_packet(bad_packet);

        assert!(matches!(
            result,
            Err(crate::error::Error::Reassembly(ReassemblyError::TotalPacketMismatch { .. }))
        ));
    }

    #[test]
    fn test_late_arrival_ignored() {
        let mut reassembler = Reassembler::new(10, 1000);

        // Complete chunk 0
        reassembler.insert_packet(make_packet(0, 0, 1)).unwrap();

        // Send late packet for chunk 0 (already emitted)
        let result = reassembler.insert_packet(make_packet(0, 0, 1));

        // Should be ignored (Ok(None)), not error
        assert!(result.is_ok());
        assert!(result.unwrap().is_none());
    }
}
