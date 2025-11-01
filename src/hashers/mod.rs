//! Hasher implementations for TS3 security level calculation

pub mod cpu;
pub mod cuda;

pub use cpu::CpuHasher;
pub use cuda::CudaHasher;
