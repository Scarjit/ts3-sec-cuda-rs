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
    },
}

#[derive(Copy, Clone, PartialEq, Eq, PartialOrd, Ord, ValueEnum)]
pub enum HasherMethod {
    /// CPU-based SHA-1 hashing
    Cpu,
    /// CUDA/GPU-based SHA-1 hashing (not yet implemented)
    Cuda,
}
