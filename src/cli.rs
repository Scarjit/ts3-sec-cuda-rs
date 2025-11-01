//! Command-line interface definitions

use clap::{Parser, Subcommand, ValueEnum};

#[derive(Parser)]
#[command(name = "ts3-sec-cuda-rs")]
#[command(about = "TeamSpeak 3 Security Level Tool", long_about = None)]
pub struct Cli {
    #[command(subcommand)]
    pub command: Command,
}

#[derive(Subcommand)]
pub enum Command {
    /// Decode and display identity information
    Decode {
        /// Path to identity.ini file
        #[arg(short, long, group = "input")]
        file: Option<String>,

        /// Identity string directly (format: "counter" + "V" + base64_key)
        #[arg(short, long, group = "input")]
        string: Option<String>,
    },
    /// Increase security level of an identity
    Increase {
        /// Path to identity.ini file
        #[arg(short, long, group = "input")]
        file: Option<String>,

        /// Identity string directly (format: "counter" + "V" + base64_key)
        #[arg(short, long, group = "input")]
        string: Option<String>,

        /// Target security level to reach
        #[arg(short, long)]
        target: u8,

        /// Hasher method to use
        #[arg(short = 'm', long, value_enum, default_value_t = HasherMethod::Cpu)]
        method: HasherMethod,

        /// Batch size for processing (CPU: 10000, CUDA: 100000)
        /// Higher values = better GPU utilization but more memory
        #[arg(short, long)]
        batch_size: Option<usize>,

        /// CUDA threads per block (32, 64, 128, 256, 512, 1024) [default: 128]
        /// 128 provides best overall performance based on benchmarks
        #[arg(long)]
        cuda_threads: Option<usize>,

        /// CUDA dynamic shared memory in bytes [default: threads_per_block * 128]
        /// Only needed if you want to override the automatic calculation
        #[arg(long)]
        cuda_shared_mem: Option<usize>,
    },
}

#[derive(Copy, Clone, PartialEq, Eq, PartialOrd, Ord, ValueEnum)]
pub enum HasherMethod {
    /// CPU-based SHA-1 hashing
    Cpu,
    /// CUDA/GPU-based SHA-1 hashing
    Cuda,
}
