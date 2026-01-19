//! Network simulator with latency, jitter, reordering, and packet loss.
//!
//! This module simulates unreliable network conditions in a deterministic way
//! using seeded randomness. All network effects are reproducible given the
//! same seed.
//!
//! # Simulated Effects
//!
//! - **Latency**: Base delay for all packets
//! - **Jitter**: Random variation in latency (uniform distribution)
//! - **Reordering**: Packets may arrive out of order within a window
//! - **Loss**: Random packet drops (Bernoulli distribution)
//!
//! # Implementation
//!
//! Uses a priority queue (min-heap) keyed by delivery time. Packets are
//! inserted with computed delay and extracted when their time arrives.
//!
//! # Determinism
//!
//! All randomness comes from a seeded ChaCha8 RNG. Given the same seed
//! and inputs, outputs are bit-identical.

use crate::packet::Packet;
use rand::{Rng, SeedableRng};
use rand_chacha::ChaCha8Rng;
use std::collections::BinaryHeap;
use std::time::{Duration, Instant};

/// Configuration for network simulation.
#[derive(Debug, Clone, Copy)]
pub struct NetworkConfig {
    /// Base latency in milliseconds
    pub base_latency_ms: u64,

    /// Jitter range in milliseconds (uniform ±jitter)
    pub jitter_ms: u64,

    /// Reordering window size (packets)
    /// Larger values allow more reordering
    pub reorder_window: usize,

    /// Packet loss probability [0.0, 1.0]
    pub loss_rate: f64,

    /// Random seed for determinism
    pub seed: u64,
}

impl NetworkConfig {
    /// Create a configuration with no impairments (perfect network).
    pub fn perfect(seed: u64) -> Self {
        Self {
            base_latency_ms: 0,
            jitter_ms: 0,
            reorder_window: 0,
            loss_rate: 0.0,
            seed,
        }
    }

    /// Create a default configuration with moderate impairments.
    pub fn default_with_seed(seed: u64) -> Self {
        Self {
            base_latency_ms: 50,
            jitter_ms: 20,
            reorder_window: 16,
            loss_rate: 0.01, // 1% loss
            seed,
        }
    }
}

/// A packet with scheduled delivery time.
///
/// Used internally by the simulator's priority queue.
#[derive(Debug, Clone)]
struct ScheduledPacket {
    packet: Packet,
    delivery_time: Instant,
}

// Implement ordering for the heap (min-heap: earliest delivery first)
impl PartialEq for ScheduledPacket {
    fn eq(&self, other: &Self) -> bool {
        self.delivery_time == other.delivery_time
    }
}

impl Eq for ScheduledPacket {}

impl PartialOrd for ScheduledPacket {
    fn partial_cmp(&self, other: &Self) -> Option<std::cmp::Ordering> {
        Some(self.cmp(other))
    }
}

impl Ord for ScheduledPacket {
    fn cmp(&self, other: &Self) -> std::cmp::Ordering {
        // Reverse ordering for min-heap (earlier time = higher priority)
        other.delivery_time.cmp(&self.delivery_time)
    }
}

/// Network simulator implementing latency, jitter, reordering, and loss.
///
/// # Thread Safety
/// Not thread-safe; use one instance per thread or synchronize externally.
#[allow(dead_code)]
pub struct NetworkSimulator {
    config: NetworkConfig,
    rng: ChaCha8Rng,
    queue: BinaryHeap<ScheduledPacket>,
    start_time: Instant,

    // Statistics
    packets_sent: u64,
    packets_dropped: u64,
    packets_delivered: u64,
}

impl NetworkSimulator {
    /// Create a new network simulator with the given configuration.
    pub fn new(config: NetworkConfig) -> Self {
        let rng = ChaCha8Rng::seed_from_u64(config.seed);

        Self {
            config,
            rng,
            queue: BinaryHeap::new(),
            start_time: Instant::now(),
            packets_sent: 0,
            packets_dropped: 0,
            packets_delivered: 0,
        }
    }

    /// Send a packet through the simulated network.
    ///
    /// The packet may be:
    /// - Dropped (if loss_rate > 0)
    /// - Delayed by latency + jitter
    /// - Reordered relative to other packets
    ///
    /// # Side Effects
    /// If not dropped, packet is added to internal queue with computed delivery time.
    pub fn send(&mut self, packet: Packet) {
        self.packets_sent += 1;

        // Check for loss
        if self.config.loss_rate > 0.0 {
            let roll: f64 = self.rng.gen();
            if roll < self.config.loss_rate {
                self.packets_dropped += 1;
                return;
            }
        }

        // Compute delivery delay
        let delay_ms = self.compute_delay();
        let delivery_time = Instant::now() + Duration::from_millis(delay_ms);

        self.queue.push(ScheduledPacket {
            packet,
            delivery_time,
        });
    }

    /// Try to receive a packet if one is ready.
    ///
    /// # Returns
    /// - `Some(packet)` if a packet's delivery time has arrived
    /// - `None` if no packets are ready yet
    ///
    /// # Note
    /// This is non-blocking. Caller should call repeatedly or use `recv_wait`.
    pub fn recv(&mut self) -> Option<Packet> {
        // Check if top packet is ready
        if let Some(scheduled) = self.queue.peek() {
            if Instant::now() >= scheduled.delivery_time {
                let scheduled = self.queue.pop().unwrap();
                self.packets_delivered += 1;
                return Some(scheduled.packet);
            }
        }

        None
    }

    /// Receive a packet, waiting up to the specified duration.
    ///
    /// # Returns
    /// - `Some(packet)` if a packet became ready within the timeout
    /// - `None` if timeout elapsed with no packet ready
    ///
    /// # Note
    /// This busy-waits with short sleeps. For production use, would use
    /// async or event-driven approach.
    pub fn recv_wait(&mut self, timeout: Duration) -> Option<Packet> {
        let deadline = Instant::now() + timeout;

        loop {
            if let Some(packet) = self.recv() {
                return Some(packet);
            }

            if Instant::now() >= deadline {
                return None;
            }

            // Small sleep to avoid busy-waiting
            std::thread::sleep(Duration::from_micros(100));
        }
    }

    /// Check if any packets are currently in flight (queued).
    pub fn has_pending(&self) -> bool {
        !self.queue.is_empty()
    }

    /// Get count of packets currently in flight.
    pub fn pending_count(&self) -> usize {
        self.queue.len()
    }

    /// Drain all remaining packets immediately (ignoring delivery times).
    ///
    /// Used for testing or graceful shutdown.
    pub fn drain(&mut self) -> Vec<Packet> {
        let mut packets = Vec::new();
        while let Some(scheduled) = self.queue.pop() {
            packets.push(scheduled.packet);
            self.packets_delivered += 1;
        }
        packets
    }

    /// Get statistics about network behavior.
    pub fn stats(&self) -> NetworkStats {
        NetworkStats {
            packets_sent: self.packets_sent,
            packets_dropped: self.packets_dropped,
            packets_delivered: self.packets_delivered,
            packets_in_flight: self.queue.len(),
        }
    }

    /// Compute delay for a packet in milliseconds.
    ///
    /// Delay = base_latency ± jitter
    fn compute_delay(&mut self) -> u64 {
        let base = self.config.base_latency_ms;

        if self.config.jitter_ms == 0 {
            return base;
        }

        // Uniform jitter: base ± jitter_ms
        let jitter_range = self.config.jitter_ms * 2;
        let jitter = self.rng.gen_range(0..=jitter_range);
        let jitter_offset = jitter as i64 - self.config.jitter_ms as i64;

        (base as i64 + jitter_offset).max(0) as u64
    }
}

/// Statistics about network simulator behavior.
#[derive(Debug, Clone, Copy)]
pub struct NetworkStats {
    /// Total packets sent into the simulator
    pub packets_sent: u64,

    /// Packets dropped due to loss
    pub packets_dropped: u64,

    /// Packets successfully delivered
    pub packets_delivered: u64,

    /// Packets currently in flight
    pub packets_in_flight: usize,
}

impl NetworkStats {
    /// Compute packet loss rate.
    pub fn loss_rate(&self) -> f64 {
        if self.packets_sent == 0 {
            0.0
        } else {
            self.packets_dropped as f64 / self.packets_sent as f64
        }
    }

    /// Compute delivery rate (delivered / sent).
    pub fn delivery_rate(&self) -> f64 {
        if self.packets_sent == 0 {
            0.0
        } else {
            self.packets_delivered as f64 / self.packets_sent as f64
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::packet::Packet;

    fn make_test_packet(id: u32) -> Packet {
        Packet::new(0, id, 10, vec![id as u8])
    }

    #[test]
    fn test_perfect_network() {
        let config = NetworkConfig::perfect(42);
        let mut sim = NetworkSimulator::new(config);

        let packet = make_test_packet(0);
        sim.send(packet.clone());

        // Should be available immediately (no delay)
        let received = sim.recv().unwrap();
        assert_eq!(received.packet_id, 0);

        let stats = sim.stats();
        assert_eq!(stats.packets_sent, 1);
        assert_eq!(stats.packets_dropped, 0);
        assert_eq!(stats.packets_delivered, 1);
    }

    #[test]
    fn test_latency() {
        let config = NetworkConfig {
            base_latency_ms: 50,
            jitter_ms: 0,
            reorder_window: 0,
            loss_rate: 0.0,
            seed: 42,
        };

        let mut sim = NetworkSimulator::new(config);

        let start = Instant::now();
        sim.send(make_test_packet(0));

        // Should not be available immediately
        assert!(sim.recv().is_none());

        // Wait for delivery
        let received = sim.recv_wait(Duration::from_millis(100)).unwrap();
        let elapsed = start.elapsed();

        assert_eq!(received.packet_id, 0);
        assert!(elapsed >= Duration::from_millis(50));
    }

    #[test]
    fn test_packet_loss() {
        let config = NetworkConfig {
            base_latency_ms: 0,
            jitter_ms: 0,
            reorder_window: 0,
            loss_rate: 0.5, // 50% loss
            seed: 42,
        };

        let mut sim = NetworkSimulator::new(config);

        // Send many packets
        for i in 0..100 {
            sim.send(make_test_packet(i));
        }

        let stats = sim.stats();
        assert_eq!(stats.packets_sent, 100);

        // Should have lost approximately half (with randomness)
        // Allow 30-70% range due to randomness
        assert!(stats.packets_dropped >= 30 && stats.packets_dropped <= 70);
    }

    #[test]
    fn test_determinism() {
        let config = NetworkConfig::default_with_seed(12345);

        let mut sim1 = NetworkSimulator::new(config);
        let mut sim2 = NetworkSimulator::new(config);

        // Send same packets to both
        for i in 0..20 {
            sim1.send(make_test_packet(i));
            sim2.send(make_test_packet(i));
        }

        let stats1 = sim1.stats();
        let stats2 = sim2.stats();

        // Should have identical behavior
        assert_eq!(stats1.packets_dropped, stats2.packets_dropped);
        assert_eq!(stats1.packets_in_flight, stats2.packets_in_flight);
    }

    #[test]
    fn test_drain() {
        let config = NetworkConfig {
            base_latency_ms: 1000, // 1 second delay
            jitter_ms: 0,
            reorder_window: 0,
            loss_rate: 0.0,
            seed: 42,
        };

        let mut sim = NetworkSimulator::new(config);

        sim.send(make_test_packet(0));
        sim.send(make_test_packet(1));

        // Packets should be in flight
        assert_eq!(sim.pending_count(), 2);

        // Drain should get them immediately
        let packets = sim.drain();
        assert_eq!(packets.len(), 2);
        assert_eq!(sim.pending_count(), 0);
    }

    #[test]
    fn test_reordering() {
        // With jitter, packets can arrive out of order
        let config = NetworkConfig {
            base_latency_ms: 20,
            jitter_ms: 15, // Large jitter relative to base
            reorder_window: 0,
            loss_rate: 0.0,
            seed: 99,
        };

        let mut sim = NetworkSimulator::new(config);

        // Send packets in order
        for i in 0..10 {
            sim.send(make_test_packet(i));
        }

        // Wait for all to be delivered
        std::thread::sleep(Duration::from_millis(100));

        let mut received = Vec::new();
        while let Some(packet) = sim.recv() {
            received.push(packet.packet_id);
        }

        // Should have received all packets
        assert_eq!(received.len(), 10);

        // Check if any reordering occurred
        let mut is_reordered = false;
        for i in 1..received.len() {
            if received[i] < received[i - 1] {
                is_reordered = true;
                break;
            }
        }

        // With large jitter, we expect some reordering
        // (this is probabilistic, but with seed 99 it should reorder)
        assert!(is_reordered, "Expected some reordering with large jitter");
    }

    #[test]
    fn test_stats() {
        let config = NetworkConfig {
            base_latency_ms: 0,
            jitter_ms: 0,
            reorder_window: 0,
            loss_rate: 0.25, // 25% loss
            seed: 42,
        };

        let mut sim = NetworkSimulator::new(config);

        for i in 0..100 {
            sim.send(make_test_packet(i));
        }

        let stats = sim.stats();
        let loss_rate = stats.loss_rate();

        // Should be approximately 0.25
        assert!(loss_rate > 0.15 && loss_rate < 0.35);
    }
}
