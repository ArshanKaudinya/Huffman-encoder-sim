//! Metrics collection and reporting for the transfer system.
//!
//! This module provides observable insights into system behavior:
//! - Throughput (bytes in/out)
//! - Compression ratio
//! - Packet-level statistics (sent, dropped, reordered)
//! - Timing information
//!
//! # Design
//!
//! Metrics are collected in a simple struct with atomic or synchronized updates.
//! For this educational system, we use a single-threaded approach with explicit
//! updates at each pipeline stage.
//!
//! # Thread Safety
//!
//! The `Metrics` struct is NOT thread-safe. For multi-threaded use, wrap in
//! `Arc<Mutex<Metrics>>` or use per-thread metrics that are merged at the end.

use std::time::{Duration, Instant};

/// Comprehensive metrics for the file transfer system.
///
/// Tracks counts, bytes, and timing across all pipeline stages.
#[derive(Debug, Clone)]
pub struct Metrics {
    // === Timing ===
    /// When the transfer started
    pub start_time: Instant,

    /// When the transfer ended (set on completion)
    pub end_time: Option<Instant>,

    // === Input/Output ===
    /// Total bytes read from input file
    pub input_bytes: u64,

    /// Total bytes written to output file
    pub output_bytes: u64,

    // === Chunking ===
    /// Number of chunks created
    pub chunks_created: u64,

    /// Total raw bytes across all chunks
    pub raw_chunk_bytes: u64,

    /// Total compressed bytes across all chunks
    pub compressed_chunk_bytes: u64,

    // === Packetization ===
    /// Total packets generated
    pub packets_generated: u64,

    // === Network ===
    /// Packets sent into network simulator
    pub packets_sent: u64,

    /// Packets dropped by simulator (loss)
    pub packets_dropped: u64,

    /// Packets received by reassembler
    pub packets_received: u64,

    /// Duplicate packets received
    pub packets_duplicate: u64,

    /// Invalid packets (bad headers, etc.)
    pub packets_invalid: u64,

    /// Packets that arrived out of order
    pub packets_reordered: u64,

    // === Reassembly ===
    /// Chunks successfully reassembled
    pub chunks_reassembled: u64,

    /// Chunks that failed CRC validation
    pub chunks_failed_crc: u64,

    /// Chunks that timed out
    pub chunks_timed_out: u64,

    // === Decoding ===
    /// Chunks successfully decoded
    pub chunks_decoded: u64,
}

impl Metrics {
    /// Create new metrics with start time set to now.
    pub fn new() -> Self {
        Self {
            start_time: Instant::now(),
            end_time: None,
            input_bytes: 0,
            output_bytes: 0,
            chunks_created: 0,
            raw_chunk_bytes: 0,
            compressed_chunk_bytes: 0,
            packets_generated: 0,
            packets_sent: 0,
            packets_dropped: 0,
            packets_received: 0,
            packets_duplicate: 0,
            packets_invalid: 0,
            packets_reordered: 0,
            chunks_reassembled: 0,
            chunks_failed_crc: 0,
            chunks_timed_out: 0,
            chunks_decoded: 0,
        }
    }

    /// Mark the transfer as complete.
    pub fn complete(&mut self) {
        self.end_time = Some(Instant::now());
    }

    /// Get total duration (or current elapsed if not complete).
    pub fn duration(&self) -> Duration {
        match self.end_time {
            Some(end) => end.duration_since(self.start_time),
            None => self.start_time.elapsed(),
        }
    }

    /// Compute compression ratio (compressed / raw).
    ///
    /// Returns 0.0 if no data compressed.
    pub fn compression_ratio(&self) -> f64 {
        if self.raw_chunk_bytes == 0 {
            0.0
        } else {
            self.compressed_chunk_bytes as f64 / self.raw_chunk_bytes as f64
        }
    }

    /// Compute packet loss rate (dropped / sent).
    pub fn loss_rate(&self) -> f64 {
        if self.packets_sent == 0 {
            0.0
        } else {
            self.packets_dropped as f64 / self.packets_sent as f64
        }
    }

    /// Compute packet reorder rate (reordered / received).
    pub fn reorder_rate(&self) -> f64 {
        if self.packets_received == 0 {
            0.0
        } else {
            self.packets_reordered as f64 / self.packets_received as f64
        }
    }

    /// Compute throughput in bytes/second.
    pub fn throughput_bps(&self) -> f64 {
        let duration_secs = self.duration().as_secs_f64();
        if duration_secs == 0.0 {
            0.0
        } else {
            self.input_bytes as f64 / duration_secs
        }
    }

    /// Print a human-readable summary to stdout.
    pub fn print_summary(&self) {
        let duration_ms = self.duration().as_millis();

        println!("\n=== Transfer Summary ===");
        println!("Duration: {} ms", duration_ms);
        println!();

        // Input/Output
        println!("Input:  {} bytes ({:.2} MiB)", self.input_bytes, self.input_bytes as f64 / 1024.0 / 1024.0);
        println!("Output: {} bytes ({:.2} MiB)", self.output_bytes, self.output_bytes as f64 / 1024.0 / 1024.0);

        // Verify match
        if self.input_bytes == self.output_bytes {
            println!("Verification: PASSED ✓");
        } else {
            println!("Verification: FAILED ✗ (size mismatch)");
        }
        println!();

        // Compression
        println!("=== Compression ===");
        println!("Chunks: {}", self.chunks_created);
        println!("Raw bytes: {} ({:.2} MiB)", self.raw_chunk_bytes, self.raw_chunk_bytes as f64 / 1024.0 / 1024.0);
        println!("Compressed: {} ({:.2} MiB)", self.compressed_chunk_bytes, self.compressed_chunk_bytes as f64 / 1024.0 / 1024.0);
        println!("Ratio: {:.1}%", self.compression_ratio() * 100.0);
        println!();

        // Network
        println!("=== Network ===");
        println!("Packets generated: {}", self.packets_generated);
        println!("Packets sent: {}", self.packets_sent);
        println!("Packets dropped: {} ({:.2}%)", self.packets_dropped, self.loss_rate() * 100.0);
        println!("Packets received: {}", self.packets_received);
        println!("Packets reordered: {} ({:.2}%)", self.packets_reordered, self.reorder_rate() * 100.0);
        println!("Packets duplicate: {}", self.packets_duplicate);
        println!("Packets invalid: {}", self.packets_invalid);
        println!();

        // Reassembly
        println!("=== Reassembly ===");
        println!("Chunks reassembled: {}", self.chunks_reassembled);
        println!("Chunks decoded: {}", self.chunks_decoded);
        println!("CRC failures: {}", self.chunks_failed_crc);
        println!("Timeouts: {}", self.chunks_timed_out);
        println!();

        // Performance
        println!("=== Performance ===");
        println!("Throughput: {:.2} MB/s", self.throughput_bps() / 1_000_000.0);
        println!();
    }

    /// Print just the final result (pass/fail).
    pub fn print_result(&self) {
        if self.input_bytes == self.output_bytes && self.chunks_timed_out == 0 && self.chunks_failed_crc == 0 {
            println!("✓ Transfer completed successfully");
            println!("  {} bytes transferred in {} ms", self.input_bytes, self.duration().as_millis());
        } else if self.chunks_timed_out > 0 {
            println!("✗ Transfer failed: {} chunks timed out", self.chunks_timed_out);
        } else if self.chunks_failed_crc > 0 {
            println!("✗ Transfer failed: {} chunks failed CRC", self.chunks_failed_crc);
        } else {
            println!("✗ Transfer failed: size mismatch ({} != {})", self.input_bytes, self.output_bytes);
        }
    }

    /// Export metrics as a simple text format (for parsing/testing).
    pub fn export_text(&self) -> String {
        format!(
            "duration_ms={}\n\
             input_bytes={}\n\
             output_bytes={}\n\
             chunks_created={}\n\
             compression_ratio={:.4}\n\
             packets_sent={}\n\
             packets_dropped={}\n\
             loss_rate={:.4}\n\
             packets_reordered={}\n\
             reorder_rate={:.4}\n\
             chunks_timed_out={}\n\
             chunks_failed_crc={}\n",
            self.duration().as_millis(),
            self.input_bytes,
            self.output_bytes,
            self.chunks_created,
            self.compression_ratio(),
            self.packets_sent,
            self.packets_dropped,
            self.loss_rate(),
            self.packets_reordered,
            self.reorder_rate(),
            self.chunks_timed_out,
            self.chunks_failed_crc,
        )
    }
}

impl Default for Metrics {
    fn default() -> Self {
        Self::new()
    }
}

/// Helper for tracking reordering.
///
/// Tracks the last packet_id seen for each chunk to detect out-of-order arrival.
pub struct ReorderTracker {
    /// For each chunk_id, track the highest packet_id seen so far
    last_packet_ids: std::collections::HashMap<u64, u32>,
}

impl ReorderTracker {
    /// Create a new reorder tracker.
    pub fn new() -> Self {
        Self {
            last_packet_ids: std::collections::HashMap::new(),
        }
    }

    /// Record a packet and return true if it arrived out of order.
    ///
    /// A packet is considered reordered if its packet_id is less than
    /// the highest packet_id we've seen for this chunk.
    pub fn track(&mut self, chunk_id: u64, packet_id: u32) -> bool {
        let last = self.last_packet_ids.entry(chunk_id).or_insert(0);

        let is_reordered = packet_id < *last;

        if packet_id > *last {
            *last = packet_id;
        }

        is_reordered
    }

    /// Clear tracking for a chunk (when it completes).
    pub fn clear_chunk(&mut self, chunk_id: u64) {
        self.last_packet_ids.remove(&chunk_id);
    }
}

impl Default for ReorderTracker {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_metrics_creation() {
        let metrics = Metrics::new();
        assert!(metrics.end_time.is_none());
        assert!(metrics.duration().as_millis() < 100); // Should be recent
    }

    #[test]
    fn test_compression_ratio() {
        let mut metrics = Metrics::new();
        metrics.raw_chunk_bytes = 1000;
        metrics.compressed_chunk_bytes = 750;

        assert_eq!(metrics.compression_ratio(), 0.75);
    }

    #[test]
    fn test_loss_rate() {
        let mut metrics = Metrics::new();
        metrics.packets_sent = 100;
        metrics.packets_dropped = 5;

        assert_eq!(metrics.loss_rate(), 0.05);
    }

    #[test]
    fn test_reorder_tracker() {
        let mut tracker = ReorderTracker::new();

        // In-order packets
        assert!(!tracker.track(0, 0));
        assert!(!tracker.track(0, 1));
        assert!(!tracker.track(0, 2));

        // Out-of-order packet
        assert!(tracker.track(0, 1)); // Already saw 2

        // Different chunk
        assert!(!tracker.track(1, 0));
        assert!(!tracker.track(1, 5));
        assert!(tracker.track(1, 3)); // Reordered within chunk 1
    }

    #[test]
    fn test_reorder_tracker_clear() {
        let mut tracker = ReorderTracker::new();

        tracker.track(0, 5);
        tracker.clear_chunk(0);

        // After clear, should not detect as reordered
        assert!(!tracker.track(0, 0));
    }

    #[test]
    fn test_throughput() {
        let mut metrics = Metrics::new();
        metrics.input_bytes = 1_000_000;

        // Simulate 1 second elapsed
        std::thread::sleep(Duration::from_millis(100));
        metrics.complete();

        let throughput = metrics.throughput_bps();
        assert!(throughput > 0.0);
    }

    #[test]
    fn test_export_text() {
        let mut metrics = Metrics::new();
        metrics.input_bytes = 1000;
        metrics.output_bytes = 1000;
        metrics.chunks_created = 10;

        let text = metrics.export_text();
        assert!(text.contains("input_bytes=1000"));
        assert!(text.contains("output_bytes=1000"));
        assert!(text.contains("chunks_created=10"));
    }
}
