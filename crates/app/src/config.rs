//! Configuration for the encoder-sim application.
//!
//! Handles parsing command-line arguments and generating sensible defaults
//! (including randomized defaults that are reproducible with a seed).
//!
//! # Philosophy
//!
//! The tool should work with ZERO arguments, using intelligent defaults.
//! All defaults are printed so runs are reproducible.

use encoder_sim_core::network::NetworkConfig;
use rand::{Rng, SeedableRng};
use rand_chacha::ChaCha8Rng;
use std::path::PathBuf;

/// Complete configuration for a transfer run.
#[derive(Debug, Clone)]
pub struct Config {
    // === Files ===
    /// Input file path (None = generate sample)
    pub input_file: Option<PathBuf>,

    /// Output file path
    pub output_file: PathBuf,

    // === Chunking ===
    /// Chunk size in bytes
    pub chunk_bytes: usize,

    // === Network ===
    /// MTU size in bytes
    pub mtu_bytes: usize,

    /// Network simulation config
    pub network: NetworkConfig,

    // === Reassembly ===
    /// Maximum chunks in flight
    pub max_inflight_chunks: usize,

    /// Reassembly timeout in milliseconds
    pub reassembly_timeout_ms: u64,

    // === Channels ===
    /// Channel capacity for pipeline stages
    pub channel_capacity: usize,

    // === Behavior ===
    /// Whether to print detailed config
    pub print_config: bool,

    /// Whether to print detailed metrics summary
    pub print_metrics: bool,
}

impl Config {
    /// Parse configuration from command-line arguments.
    ///
    /// If no arguments provided, generates randomized defaults using a time-based seed.
    /// If --seed is provided, uses that seed for all randomness (fully deterministic).
    pub fn from_args(args: &[String]) -> Result<Self, String> {
        let mut input_file: Option<PathBuf> = None;
        let mut output_file: Option<PathBuf> = None;
        let mut seed: Option<u64> = None;
        let mut chunk_bytes: Option<usize> = None;
        let mut mtu_bytes: Option<usize> = None;
        let mut base_latency_ms: Option<u64> = None;
        let mut jitter_ms: Option<u64> = None;
        let mut reorder_window: Option<usize> = None;
        let mut loss_rate: Option<f64> = None;
        let mut reassembly_timeout_ms: Option<u64> = None;
        let mut max_inflight_chunks: Option<usize> = None;
        let mut channel_capacity: Option<usize> = None;
        let mut print_config = false;
        let mut print_metrics = true;

        let mut i = 0;
        while i < args.len() {
            match args[i].as_str() {
                "--in" => {
                    i += 1;
                    if i >= args.len() {
                        return Err("--in requires a path".to_string());
                    }
                    input_file = Some(PathBuf::from(&args[i]));
                }
                "--out" => {
                    i += 1;
                    if i >= args.len() {
                        return Err("--out requires a path".to_string());
                    }
                    output_file = Some(PathBuf::from(&args[i]));
                }
                "--seed" => {
                    i += 1;
                    if i >= args.len() {
                        return Err("--seed requires a number".to_string());
                    }
                    seed = Some(args[i].parse().map_err(|_| "invalid seed")?);
                }
                "--chunk-bytes" => {
                    i += 1;
                    if i >= args.len() {
                        return Err("--chunk-bytes requires a number".to_string());
                    }
                    chunk_bytes = Some(args[i].parse().map_err(|_| "invalid chunk-bytes")?);
                }
                "--mtu" => {
                    i += 1;
                    if i >= args.len() {
                        return Err("--mtu requires a number".to_string());
                    }
                    mtu_bytes = Some(args[i].parse().map_err(|_| "invalid mtu")?);
                }
                "--latency" => {
                    i += 1;
                    if i >= args.len() {
                        return Err("--latency requires a number".to_string());
                    }
                    base_latency_ms = Some(args[i].parse().map_err(|_| "invalid latency")?);
                }
                "--jitter" => {
                    i += 1;
                    if i >= args.len() {
                        return Err("--jitter requires a number".to_string());
                    }
                    jitter_ms = Some(args[i].parse().map_err(|_| "invalid jitter")?);
                }
                "--reorder-window" => {
                    i += 1;
                    if i >= args.len() {
                        return Err("--reorder-window requires a number".to_string());
                    }
                    reorder_window = Some(args[i].parse().map_err(|_| "invalid reorder-window")?);
                }
                "--loss" => {
                    i += 1;
                    if i >= args.len() {
                        return Err("--loss requires a number".to_string());
                    }
                    loss_rate = Some(args[i].parse().map_err(|_| "invalid loss rate")?);
                }
                "--no-loss" => {
                    loss_rate = Some(0.0);
                }
                "--timeout" => {
                    i += 1;
                    if i >= args.len() {
                        return Err("--timeout requires a number".to_string());
                    }
                    reassembly_timeout_ms = Some(args[i].parse().map_err(|_| "invalid timeout")?);
                }
                "--max-inflight" => {
                    i += 1;
                    if i >= args.len() {
                        return Err("--max-inflight requires a number".to_string());
                    }
                    max_inflight_chunks = Some(args[i].parse().map_err(|_| "invalid max-inflight")?);
                }
                "--channel-capacity" => {
                    i += 1;
                    if i >= args.len() {
                        return Err("--channel-capacity requires a number".to_string());
                    }
                    channel_capacity = Some(args[i].parse().map_err(|_| "invalid channel-capacity")?);
                }
                "--print-config" => {
                    print_config = true;
                }
                "--no-metrics" => {
                    print_metrics = false;
                }
                "--help" | "-h" => {
                    print_help();
                    std::process::exit(0);
                }
                _ => {
                    return Err(format!("unknown argument: {}", args[i]));
                }
            }
            i += 1;
        }

        // Determine seed (explicit or time-based)
        let seed = seed.unwrap_or_else(|| {
            use std::time::{SystemTime, UNIX_EPOCH};
            let t = SystemTime::now()
                .duration_since(UNIX_EPOCH)
                .unwrap()
                .as_millis() as u64;
            t
        });

        // Generate defaults using seed
        let mut rng = ChaCha8Rng::seed_from_u64(seed);

        let config = Config {
            input_file,
            output_file: output_file.unwrap_or_else(|| PathBuf::from("./out.bin")),
            chunk_bytes: chunk_bytes.unwrap_or(65536), // 64 KiB
            mtu_bytes: mtu_bytes.unwrap_or(1200),
            network: NetworkConfig {
                base_latency_ms: base_latency_ms.unwrap_or_else(|| rng.gen_range(10..=80)),
                jitter_ms: jitter_ms.unwrap_or_else(|| rng.gen_range(0..=40)),
                reorder_window: reorder_window.unwrap_or_else(|| rng.gen_range(0..=64)),
                loss_rate: loss_rate.unwrap_or_else(|| {
                    // Bias toward small loss rates
                    let r: f64 = rng.gen();
                    (r * r * 0.05).min(0.05) // 0-5%, biased toward 0
                }),
                seed,
            },
            max_inflight_chunks: max_inflight_chunks.unwrap_or(64),
            reassembly_timeout_ms: reassembly_timeout_ms.unwrap_or_else(|| rng.gen_range(500..=3000)),
            channel_capacity: channel_capacity.unwrap_or(32),
            print_config,
            print_metrics,
        };

        Ok(config)
    }

    /// Print the configuration in human-readable form.
    pub fn print(&self) {
        println!("=== Configuration ===");
        println!("Input file:  {:?}", self.input_file.as_ref().map_or("(generate sample)", |p| p.to_str().unwrap()));
        println!("Output file: {:?}", self.output_file.to_str().unwrap());
        println!();
        println!("Chunk size: {} bytes ({} KiB)", self.chunk_bytes, self.chunk_bytes / 1024);
        println!("MTU: {} bytes", self.mtu_bytes);
        println!();
        println!("=== Network Simulation ===");
        println!("Seed: {}", self.network.seed);
        println!("Base latency: {} ms", self.network.base_latency_ms);
        println!("Jitter: ±{} ms", self.network.jitter_ms);
        println!("Reorder window: {} packets", self.network.reorder_window);
        println!("Loss rate: {:.2}%", self.network.loss_rate * 100.0);
        println!();
        println!("=== Reassembly ===");
        println!("Max in-flight chunks: {}", self.max_inflight_chunks);
        println!("Timeout: {} ms", self.reassembly_timeout_ms);
        println!("Channel capacity: {}", self.channel_capacity);
        println!();
    }
}

fn print_help() {
    println!("encoder-sim: Educational file transfer with Huffman compression");
    println!();
    println!("USAGE:");
    println!("    encoder-sim [OPTIONS]");
    println!();
    println!("OPTIONS:");
    println!("    --in <PATH>             Input file (default: generate sample)");
    println!("    --out <PATH>            Output file (default: ./out.bin)");
    println!("    --seed <N>              Random seed for determinism");
    println!();
    println!("    --chunk-bytes <N>       Chunk size (default: 65536)");
    println!("    --mtu <N>               MTU size (default: 1200)");
    println!();
    println!("    --latency <MS>          Base network latency (default: random 10-80)");
    println!("    --jitter <MS>           Latency jitter (default: random 0-40)");
    println!("    --reorder-window <N>    Reorder window (default: random 0-64)");
    println!("    --loss <RATE>           Packet loss rate 0.0-1.0 (default: random 0-0.05)");
    println!("    --no-loss               Disable packet loss (same as --loss 0)");
    println!("    --timeout <MS>          Reassembly timeout (default: random 500-3000)");
    println!();
    println!("    --max-inflight <N>      Max chunks in flight (default: 64)");
    println!("    --channel-capacity <N>  Pipeline channel capacity (default: 32)");
    println!();
    println!("    --print-config          Print resolved configuration");
    println!("    --no-metrics            Don't print metrics summary");
    println!("    --help, -h              Print this help");
    println!();
    println!("EXAMPLES:");
    println!("    encoder-sim                                    # Run with random defaults");
    println!("    encoder-sim --seed 42                          # Deterministic run");
    println!("    encoder-sim --in file.bin --out received.bin  # Transfer specific file");
    println!("    encoder-sim --no-loss --latency 10             # Perfect network, 10ms delay");
    println!();
}
