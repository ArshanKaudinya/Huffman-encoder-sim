//! Input file generation for testing.
//!
//! When no input file is specified, we generate a sample file with interesting
//! compression characteristics: mix of repetitive and random data.
//!
//! # Design
//!
//! Generated data has:
//! - Some highly compressible sections (runs of same byte)
//! - Some moderately compressible sections (text-like data)
//! - Some incompressible sections (random bytes)
//!
//! This makes the compression behavior visible in metrics.

use rand::{Rng, SeedableRng};
use rand_chacha::ChaCha8Rng;
use std::io::Write;

/// Generate a sample input file with mixed compressibility.
///
/// # Arguments
/// - `seed`: random seed for determinism
/// - `size_bytes`: approximate size of generated data
///
/// # Returns
/// Vector of bytes ready to be written to file or processed.
pub fn generate_sample_data(seed: u64, size_bytes: usize) -> Vec<u8> {
    let mut rng = ChaCha8Rng::seed_from_u64(seed);
    let mut data = Vec::with_capacity(size_bytes);

    // Generate data in chunks with different compressibility
    let mut remaining = size_bytes;

    while remaining > 0 {
        let chunk_size = remaining.min(8192);

        // Choose chunk type randomly
        let chunk_type: u8 = rng.gen_range(0..10);

        match chunk_type {
            // 30% highly compressible (runs of same byte)
            0..=2 => {
                let byte_value: u8 = rng.gen();
                data.extend(std::iter::repeat(byte_value).take(chunk_size));
            }

            // 30% moderately compressible (limited alphabet, text-like)
            3..=5 => {
                let alphabet = b"abcdefghijklmnopqrstuvwxyz .!,\n";
                for _ in 0..chunk_size {
                    let idx = rng.gen_range(0..alphabet.len());
                    data.push(alphabet[idx]);
                }
            }

            // 20% structured (repeating patterns)
            6..=7 => {
                let pattern = generate_pattern(&mut rng);
                let mut pos = 0;
                for _ in 0..chunk_size {
                    data.push(pattern[pos % pattern.len()]);
                    pos += 1;
                }
            }

            // 20% incompressible (random bytes)
            _ => {
                for _ in 0..chunk_size {
                    data.push(rng.gen());
                }
            }
        }

        remaining = remaining.saturating_sub(chunk_size);
    }

    // Truncate to exact size
    data.truncate(size_bytes);
    data
}

/// Generate a small repeating pattern.
fn generate_pattern(rng: &mut ChaCha8Rng) -> Vec<u8> {
    let pattern_len = rng.gen_range(4..=32);
    (0..pattern_len).map(|_| rng.gen()).collect()
}

/// Write generated data to a file.
pub fn write_sample_file(path: &std::path::Path, seed: u64, size_bytes: usize) -> std::io::Result<()> {
    let data = generate_sample_data(seed, size_bytes);
    let mut file = std::fs::File::create(path)?;
    file.write_all(&data)?;
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_generate_sample_data() {
        let data = generate_sample_data(42, 1000);
        assert_eq!(data.len(), 1000);
    }

    #[test]
    fn test_determinism() {
        let data1 = generate_sample_data(12345, 5000);
        let data2 = generate_sample_data(12345, 5000);

        assert_eq!(data1, data2);
    }

    #[test]
    fn test_different_seeds() {
        let data1 = generate_sample_data(1, 1000);
        let data2 = generate_sample_data(2, 1000);

        assert_ne!(data1, data2);
    }

    #[test]
    fn test_various_sizes() {
        for size in [0, 1, 100, 1000, 10000, 100000] {
            let data = generate_sample_data(999, size);
            assert_eq!(data.len(), size);
        }
    }
}
