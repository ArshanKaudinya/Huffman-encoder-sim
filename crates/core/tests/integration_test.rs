//! Integration tests for the full encoder-sim pipeline.
//!
//! These tests verify end-to-end behavior: input -> compress -> packetize ->
//! network -> reassemble -> decompress -> output, with verification that
//! output matches input.

use encoder_sim_core::{
    framing::{compress_and_frame, decompress_frame, parse_chunk_frame},
    network::{NetworkConfig, NetworkSimulator},
    packet::{packetize, reassemble, Packet},
    reassembly::Reassembler,
};

/// Test a simple round-trip: compress, packetize, network, reassemble, decompress.
#[test]
fn test_full_pipeline_no_loss() {
    let input_data = b"hello world! this is a test of the full pipeline with some repetition: aaaaaaaaaa bbbbbbbbbb cccccccccc";

    // Step 1: Compress and frame
    let chunk_frame = compress_and_frame(0, input_data).expect("compression failed");

    // Step 2: Packetize
    let mtu = 100;
    let packets = packetize(0, &chunk_frame, mtu).expect("packetization failed");

    println!("Generated {} packets", packets.len());

    // Step 3: Send through perfect network
    let config = NetworkConfig::perfect(42);
    let mut network = NetworkSimulator::new(config);

    for packet in packets {
        network.send(packet);
    }

    // Step 4: Receive packets
    let mut received_packets = Vec::new();
    while let Some(packet) = network.recv() {
        received_packets.push(packet);
    }

    println!("Received {} packets", received_packets.len());

    // Step 5: Reassemble
    let reassembled_frame = reassemble(&mut received_packets).expect("reassembly failed");

    // Step 6: Parse and decompress
    let frame = parse_chunk_frame(&reassembled_frame).expect("frame parsing failed");
    let decoded = decompress_frame(&frame).expect("decompression failed");

    // Verify
    assert_eq!(decoded, input_data, "output doesn't match input");
}

/// Test with network impairments (latency, jitter, reordering).
#[test]
fn test_full_pipeline_with_impairments() {
    let input_data = b"The quick brown fox jumps over the lazy dog. ".repeat(100);

    // Compress and frame
    let chunk_frame = compress_and_frame(0, &input_data).expect("compression failed");

    // Packetize
    let mtu = 200;
    let packets = packetize(0, &chunk_frame, mtu).expect("packetization failed");

    // Send through impaired network (but no loss)
    let config = NetworkConfig {
        base_latency_ms: 10,
        jitter_ms: 5,
        reorder_window: 8,
        loss_rate: 0.0, // No loss for this test
        seed: 12345,
    };

    let mut network = NetworkSimulator::new(config);

    for packet in packets {
        network.send(packet);
    }

    // Wait for delivery with timeout
    std::thread::sleep(std::time::Duration::from_millis(100));

    // Receive packets (may be reordered)
    let mut received_packets = Vec::new();
    while let Some(packet) = network.recv() {
        received_packets.push(packet);
    }

    // Reassemble
    let reassembled_frame = reassemble(&mut received_packets).expect("reassembly failed");

    // Parse and decompress
    let frame = parse_chunk_frame(&reassembled_frame).expect("frame parsing failed");
    let decoded = decompress_frame(&frame).expect("decompression failed");

    // Verify
    assert_eq!(decoded, input_data, "output doesn't match input");
}

/// Test reassembler with multiple chunks arriving out of order.
#[test]
fn test_reassembler_multiple_chunks() {
    let mut reassembler = Reassembler::new(10, 5000);

    // Create two chunks
    let data1 = b"chunk one data";
    let data2 = b"chunk two data";

    let frame1 = compress_and_frame(0, data1).unwrap();
    let frame2 = compress_and_frame(1, data2).unwrap();

    // Packetize both
    let packets1 = packetize(0, &frame1, 100).unwrap();
    let packets2 = packetize(1, &frame2, 100).unwrap();

    // Send chunk 1 packets first
    for packet in &packets2 {
        let result = reassembler.insert_packet(packet.clone()).unwrap();
        // Chunk 1 should buffer since chunk 0 isn't complete yet
        assert!(result.is_none());
    }

    // Now send chunk 0 packets
    let mut chunk0_complete = false;
    for (i, packet) in packets1.iter().enumerate() {
        let result = reassembler.insert_packet(packet.clone()).unwrap();
        if i == packets1.len() - 1 {
            // Last packet should complete chunk 0
            assert!(result.is_some());
            let (chunk_id, _) = result.unwrap();
            assert_eq!(chunk_id, 0);
            chunk0_complete = true;
        }
    }

    assert!(chunk0_complete);

    // Now drain should emit chunk 1
    let drained = reassembler.drain_completed();
    assert_eq!(drained.len(), 1);
    assert_eq!(drained[0].0, 1);
}

/// Test with all symbols present (full 256-byte alphabet).
#[test]
fn test_all_symbols() {
    // Create data with all possible byte values
    let input_data: Vec<u8> = (0..=255).collect();

    println!("Input data length: {}", input_data.len());
    let frame = compress_and_frame(0, &input_data).expect("compress_and_frame failed");
    println!("Frame created, length: {}", frame.len());

    let packets = packetize(0, &frame, 100).expect("packetize failed");
    println!("Packets created: {}", packets.len());

    let mut reassembled = reassemble(&mut packets.clone()).expect("reassemble failed");
    println!("Reassembled, length: {}", reassembled.len());

    let frame = parse_chunk_frame(&reassembled).expect("parse_chunk_frame failed");
    println!("Frame parsed");

    let decoded = decompress_frame(&frame).expect("decompress_frame failed");
    println!("Decoded, length: {}", decoded.len());

    assert_eq!(decoded, input_data);
}

/// Test with large data (multiple chunks worth).
#[test]
fn test_large_data() {
    // 128 KiB of patterned data
    let input_data = vec![b'X'; 128 * 1024];

    // Split into chunks
    let chunk_size = 64 * 1024;
    let chunks: Vec<_> = input_data.chunks(chunk_size).collect();

    let mut all_packets = Vec::new();

    // Compress and packetize each chunk
    for (i, chunk) in chunks.iter().enumerate() {
        let frame = compress_and_frame(i as u64, chunk).unwrap();
        let packets = packetize(i as u64, &frame, 1200).unwrap();
        all_packets.extend(packets);
    }

    println!("Generated {} packets for {} KiB", all_packets.len(), input_data.len() / 1024);

    // Send through perfect network
    let config = NetworkConfig::perfect(999);
    let mut network = NetworkSimulator::new(config);

    for packet in all_packets {
        network.send(packet);
    }

    // Reassemble
    let mut reassembler = Reassembler::new(10, 5000);
    let mut decoded_chunks = Vec::new();

    while let Some(packet) = network.recv() {
        if let Ok(Some((chunk_id, frame_bytes))) = reassembler.insert_packet(packet) {
            decoded_chunks.push((chunk_id, frame_bytes));

            // Drain any additional completed chunks
            for chunk in reassembler.drain_completed() {
                decoded_chunks.push(chunk);
            }
        }
    }

    // Sort by chunk_id and decode
    decoded_chunks.sort_by_key(|(id, _)| *id);

    let mut output = Vec::new();
    for (_id, frame_bytes) in decoded_chunks {
        let frame = parse_chunk_frame(&frame_bytes).unwrap();
        let decoded = decompress_frame(&frame).unwrap();
        output.extend(decoded);
    }

    assert_eq!(output.len(), input_data.len());
    assert_eq!(output, input_data);
}

/// Test CRC detection of corruption.
#[test]
fn test_crc_corruption_detection() {
    let input_data = b"test data for crc validation";

    let mut frame = compress_and_frame(0, input_data).unwrap();

    // Corrupt a byte in the payload
    let len = frame.len();
    frame[len - 1] ^= 0xFF;

    // Parsing should fail with CRC error
    let result = parse_chunk_frame(&frame);
    assert!(result.is_err());
}
