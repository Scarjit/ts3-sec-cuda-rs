//! CUDA-based SHA-1 hasher implementation for TS3 security level calculation
//!
//! This module provides GPU-accelerated hashing using NVIDIA CUDA.

use crate::level_improver::SecurityLevelHasher;
use cudarc::driver::*;
use cudarc::nvrtc::Ptx;
use std::sync::Arc;

/// CUDA-based hasher using GPU acceleration
pub struct CudaHasher {
    ctx: Arc<CudaContext>,
    module: Arc<CudaModule>,
}

impl CudaHasher {
    /// Create a new CUDA hasher
    ///
    /// # Errors
    ///
    /// Returns an error if CUDA is not available or initialization fails
    pub fn new() -> Result<Self, CudaError> {
        let ctx = CudaContext::new(0).map_err(CudaError::DeviceInitError)?;

        // Load and compile the CUDA kernel
        let ptx = compile_kernel()?;
        let module = ctx.load_module(ptx)
            .map_err(CudaError::KernelLoadError)?;

        Ok(Self { ctx, module })
    }

    /// Hash a message using the CUDA kernel
    /// Returns the 20-byte (160-bit) SHA1 hash
    pub fn hash_message(&self, message: &[u8]) -> Result<[u8; 20], CudaError> {
        // Prepare the padded input block (512 bits = 64 bytes = 16 words)
        let input_block = prepare_sha1_input(message);

        let stream = self.ctx.default_stream();

        // Allocate device memory and copy input
        let d_input = stream.memcpy_stod(&input_block)
            .map_err(CudaError::MemoryError)?;
        let mut d_output = stream.alloc_zeros::<u32>(5)
            .map_err(CudaError::MemoryError)?;

        // Load the kernel function
        let sha1_func = self.module.load_function("sha1_simple")
            .map_err(|e| CudaError::KernelLoadError(e))?;

        // Launch kernel: 1 hash, 1 thread
        let cfg = LaunchConfig {
            grid_dim: (1, 1, 1),
            block_dim: (1, 1, 1),
            shared_mem_bytes: 0,
        };

        unsafe {
            stream.launch_builder(&sha1_func)
                .arg(&d_input)
                .arg(&mut d_output)
                .arg(&1i32)
                .launch(cfg)
        }.map_err(CudaError::KernelLaunchError)?;

        // Copy result back to host
        let output: Vec<u32> = stream.memcpy_dtov(&d_output)
            .map_err(CudaError::MemoryError)?;

        // Convert from u32 words to bytes (big-endian)
        let mut hash_bytes = [0u8; 20];
        for (i, &word) in output.iter().enumerate() {
            let bytes = word.to_be_bytes();
            hash_bytes[i * 4..(i + 1) * 4].copy_from_slice(&bytes);
        }

        Ok(hash_bytes)
    }
}

impl SecurityLevelHasher for CudaHasher {
    fn calculate_level(&self, public_key: &str, counter: u64) -> u8 {
        // Concatenate public_key and counter (as strings, matching CPU implementation)
        let counter_str = counter.to_string();
        let mut message = Vec::new();
        message.extend_from_slice(public_key.as_bytes());
        message.extend_from_slice(counter_str.as_bytes());

        // Hash the message
        let hash = match self.hash_message(&message) {
            Ok(h) => h,
            Err(_) => {
                // If CUDA hashing fails, fall back to a safe default
                // In production, you might want to log this error
                return 0;
            }
        };

        // Count trailing zero bits
        crate::helpers::count_trailing_zero_bits(&hash)
    }

    fn name(&self) -> &str {
        "CUDA"
    }
}

/// Prepare a message for SHA1 hashing by adding proper padding
/// Returns a 16-word (64-byte) block ready for the CUDA kernel
///
/// Note: Currently only supports messages up to 55 bytes
/// For longer messages, multi-block processing would be needed
fn prepare_sha1_input(message: &[u8]) -> [u32; 16] {
    let mut block = [0u32; 16];
    let mut bytes = [0u8; 64];

    let msg_len = message.len();
    let bit_len = (msg_len * 8) as u64;

    // Check if message fits in a single block (max 55 bytes)
    // We need: message + 0x80 (1 byte) + length (8 bytes) <= 64 bytes
    if msg_len > 55 {
        // For messages > 55 bytes, we'd need multi-block processing
        // For now, just take the first 55 bytes
        // TODO: Implement proper multi-block SHA1
        bytes[..55].copy_from_slice(&message[..55]);
        bytes[55] = 0x80;
        let truncated_bit_len = (55 * 8) as u64;
        let len_bytes = truncated_bit_len.to_be_bytes();
        bytes[56..64].copy_from_slice(&len_bytes);
    } else {
        // Copy message
        bytes[..msg_len].copy_from_slice(message);

        // Append the '1' bit (0x80)
        bytes[msg_len] = 0x80;

        // The last 8 bytes are for the length in bits (big-endian)
        // Length goes at bytes[56..64]
        let len_bytes = bit_len.to_be_bytes();
        bytes[56..64].copy_from_slice(&len_bytes);
    }

    // Convert to u32 words (big-endian)
    for i in 0..16 {
        let word_bytes = [
            bytes[i * 4],
            bytes[i * 4 + 1],
            bytes[i * 4 + 2],
            bytes[i * 4 + 3],
        ];
        block[i] = u32::from_be_bytes(word_bytes);
    }

    block
}

/// Compile the CUDA kernel from the sha1.cu file
fn compile_kernel() -> Result<Ptx, CudaError> {
    let kernel_src = include_str!("sha1.cu");

    let ptx = cudarc::nvrtc::compile_ptx(kernel_src)
        .map_err(|e| CudaError::CompileError(e.to_string()))?;

    Ok(ptx)
}

/// Errors that can occur with CUDA operations
#[derive(Debug, thiserror::Error)]
pub enum CudaError {
    #[error("Failed to initialize CUDA device: {0}")]
    DeviceInitError(#[source] cudarc::driver::DriverError),

    #[error("Failed to compile CUDA kernel: {0}")]
    CompileError(String),

    #[error("Failed to load CUDA kernel: {0}")]
    KernelLoadError(#[source] cudarc::driver::DriverError),

    #[error("Failed to launch CUDA kernel: {0}")]
    KernelLaunchError(#[source] cudarc::driver::DriverError),

    #[error("CUDA memory error: {0}")]
    MemoryError(#[source] cudarc::driver::DriverError),
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_sha1_abc() {
        // Test that SHA1("abc") = a9993e364706816aba3e25717850c26c9cd0d89d
        let hasher = CudaHasher::new().expect("Failed to initialize CUDA hasher");

        let message = b"abc";
        let hash = hasher.hash_message(message).expect("Failed to hash message");

        // Convert hash to hex string
        let hash_hex = hash.iter()
            .map(|b| format!("{:02x}", b))
            .collect::<String>();

        let expected = "a9993e364706816aba3e25717850c26c9cd0d89d";
        assert_eq!(hash_hex, expected, "SHA1 hash mismatch!");
    }

    #[test]
    fn test_sha1_long(){
        // Test that SHA1("111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111") = 748c9948d2eeb9a30364174b518df7787625f4cb
        let hasher = CudaHasher::new().expect("Failed to initialize CUDA hasher");

        let message = b"111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111";
        let hash = hasher.hash_message(message).expect("Failed to hash message");

        // Convert hash to hex string
        let hash_hex = hash.iter()
            .map(|b| format!("{:02x}", b))
            .collect::<String>();

        let expected = "748c9948d2eeb9a30364174b518df7787625f4cb";
        assert_eq!(hash_hex, expected, "SHA1 hash mismatch!");
    }

    #[test]
    fn test_hasher_name() {
        let hasher = CudaHasher::new().expect("Failed to initialize CUDA hasher");
        assert_eq!(hasher.name(), "CUDA");
    }

    #[test]
    fn test_prepare_sha1_input() {
        // Test the padding function with "abc"
        let message = b"abc";
        let block = prepare_sha1_input(message);

        // Verify the structure: "abc" + 0x80 + padding + length
        // First word should contain "abc" + 0x80 in big-endian
        // That's: 0x61626380
        assert_eq!(block[0], 0x61626380);

        // Middle words should be zero
        for i in 1..14 {
            assert_eq!(block[i], 0, "Padding word {} should be zero", i);
        }

        // Last two words should contain the bit length (24 bits = 0x18)
        assert_eq!(block[14], 0, "Upper length word should be zero");
        assert_eq!(block[15], 24, "Lower length word should be 24 (bits)");
    }

    #[test]
    fn test_cuda_matches_cpu_short_message() {
        use crate::hashers::CpuHasher;

        // Test with a short public key and counter that fits in one block
        let public_key = "test_key_123";
        let counter = 42;

        let cuda_hasher = CudaHasher::new().expect("Failed to initialize CUDA hasher");
        let cpu_hasher = CpuHasher;

        let cuda_level = cuda_hasher.calculate_level(public_key, counter);
        let cpu_level = cpu_hasher.calculate_level(public_key, counter);

        assert_eq!(cuda_level, cpu_level,
            "CUDA and CPU implementations should produce the same result");
    }
}
