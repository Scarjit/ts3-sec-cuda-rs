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
    /// Supports multi-block messages of any length
    pub fn hash_message(&self, message: &[u8]) -> Result<[u8; 20], CudaError> {
        let blocks = prepare_sha1_blocks(message);

        let stream = self.ctx.default_stream();
        let sha1_func = self.module.load_function("sha1_simple")
            .map_err(|e| CudaError::KernelLoadError(e))?;

        // SHA1 initial hash values
        let mut h: [u32; 5] = [
            0x67452301,
            0xEFCDAB89,
            0x98BADCFE,
            0x10325476,
            0xC3D2E1F0,
        ];

        // Process each block, chaining the hash values
        for block in blocks {
            // Allocate device memory and copy input block
            let d_input = stream.memcpy_stod(&block)
                .map_err(CudaError::MemoryError)?;

            // Copy current hash state to device
            let d_init_hash = stream.memcpy_stod(&h)
                .map_err(CudaError::MemoryError)?;

            let mut d_output = stream.alloc_zeros::<u32>(5)
                .map_err(CudaError::MemoryError)?;

            // Launch kernel: 1 hash, 1 thread
            let cfg = LaunchConfig {
                grid_dim: (1, 1, 1),
                block_dim: (1, 1, 1),
                shared_mem_bytes: 0,
            };

            // SAFETY: Kernel launch is safe because:
            // 1. d_input is a valid CudaSlice containing 16 u32 values (matching kernel's const unsigned int* inputs)
            // 2. d_output is a valid CudaSlice with space for 5 u32 values (matching kernel's unsigned int* outputs)
            // 3. num_hashes is correctly set to 1 (matching kernel's int num_hashes)
            // 4. d_init_hash is a valid CudaSlice containing 5 u32 values (matching kernel's const unsigned int* init_hash)
            // 5. All memory is allocated on the same device and outlives the kernel execution
            // 6. The kernel grid/block dimensions are set to (1,1,1) ensuring only one thread executes
            unsafe {
                stream.launch_builder(&sha1_func)
                    .arg(&d_input)
                    .arg(&mut d_output)
                    .arg(&1i32)
                    .arg(&d_init_hash)
                    .launch(cfg)
            }.map_err(CudaError::KernelLaunchError)?;

            // Copy result back to host
            let output: Vec<u32> = stream.memcpy_dtov(&d_output)
                .map_err(CudaError::MemoryError)?;

            // Update hash values for next block (chaining)
            h.copy_from_slice(&output[..5]);
        }

        // Convert final hash from u32 words to bytes (big-endian)
        let mut hash_bytes = [0u8; 20];
        for (i, &word) in h.iter().enumerate() {
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

/// Prepare a message for SHA1 hashing by splitting it into 64-byte blocks
/// and applying proper SHA1 padding to the final block.
/// Returns a vector of blocks, where each block is 16 u32 words (64 bytes)
fn prepare_sha1_blocks(message: &[u8]) -> Vec<[u32; 16]> {
    let msg_len = message.len();
    let bit_len = (msg_len * 8) as u64;

    // Calculate number of blocks needed
    // Each block is 64 bytes. We need space for: message + 0x80 byte + 8-byte length
    let blocks_needed = (msg_len + 1 + 8 + 63) / 64;

    let mut blocks = Vec::with_capacity(blocks_needed);

    // Process full 64-byte blocks from the message
    let mut pos = 0;
    while pos + 64 <= msg_len {
        let mut block = [0u32; 16];
        let chunk = &message[pos..pos + 64];

        // Convert bytes to u32 words (big-endian)
        for i in 0..16 {
            let word_bytes = [
                chunk[i * 4],
                chunk[i * 4 + 1],
                chunk[i * 4 + 2],
                chunk[i * 4 + 3],
            ];
            block[i] = u32::from_be_bytes(word_bytes);
        }
        blocks.push(block);
        pos += 64;
    }

    // Handle the final block(s) with remaining data and padding
    let remaining = msg_len - pos;
    let mut final_bytes = vec![0u8; blocks_needed * 64 - pos];

    // Copy remaining message bytes
    final_bytes[..remaining].copy_from_slice(&message[pos..]);

    // Add the '1' bit (0x80 byte)
    final_bytes[remaining] = 0x80;

    // Add length in bits at the end (last 8 bytes, big-endian)
    let len_pos = final_bytes.len() - 8;
    final_bytes[len_pos..].copy_from_slice(&bit_len.to_be_bytes());

    // Convert remaining bytes to blocks
    for chunk in final_bytes.chunks(64) {
        let mut block = [0u32; 16];
        for i in 0..16 {
            let word_bytes = [
                chunk[i * 4],
                chunk[i * 4 + 1],
                chunk[i * 4 + 2],
                chunk[i * 4 + 3],
            ];
            block[i] = u32::from_be_bytes(word_bytes);
        }
        blocks.push(block);
    }

    blocks
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
        // Test that SHA1("111...") (200 '1's) = b5921a6150054ca0e86598e4864f985c6f3dec59
        let hasher = CudaHasher::new().expect("Failed to initialize CUDA hasher");

        let message = b"111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111";
        let hash = hasher.hash_message(message).expect("Failed to hash message");

        // Convert hash to hex string
        let hash_hex = hash.iter()
            .map(|b| format!("{:02x}", b))
            .collect::<String>();

        let expected = "b5921a6150054ca0e86598e4864f985c6f3dec59";
        assert_eq!(hash_hex, expected, "SHA1 hash mismatch!");
    }

    #[test]
    fn test_hasher_name() {
        let hasher = CudaHasher::new().expect("Failed to initialize CUDA hasher");
        assert_eq!(hasher.name(), "CUDA");
    }

    #[test]
    fn test_prepare_sha1_blocks() {
        // Test the padding function with "abc" (single block)
        let message = b"abc";
        let blocks = prepare_sha1_blocks(message);

        assert_eq!(blocks.len(), 1, "Short message should produce 1 block");
        let block = blocks[0];

        // Verify the structure: "abc" + 0x80 + padding + length
        // First word should contain "abc" + 0x80 in big-endian: 0x61626380
        assert_eq!(block[0], 0x61626380);

        // Middle words should be zero
        for i in 1..14 {
            assert_eq!(block[i], 0, "Padding word {} should be zero", i);
        }

        // Last two words should contain the bit length (24 bits = 0x18)
        assert_eq!(block[14], 0, "Upper length word should be zero");
        assert_eq!(block[15], 24, "Lower length word should be 24 (bits)");

        // Test with a longer message (200 bytes, should produce 4 blocks)
        let long_message = b"111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111";
        let long_blocks = prepare_sha1_blocks(long_message);

        // 200 bytes + 1 byte padding + 8 byte length = 209 bytes
        // 209 / 64 = 3.27, so we need 4 blocks
        assert_eq!(long_blocks.len(), 4, "Long message should produce 4 blocks");
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

    #[test]
    fn test_cuda_matches_cpu_long_message() {
        use crate::hashers::CpuHasher;

        // Test with a long public key that requires multi-block processing
        let public_key = "ME0DAgcAAgEgAiEAy/hhqSBja7A6FTZG5s+BMnQfCqYyS9sGsbyMKBb7spYCIQCBEtZWrZtewnxuh2hsigJswGHchu3XcaiQDZziMsxTsA==";
        let counter = 12345;

        let cuda_hasher = CudaHasher::new().expect("Failed to initialize CUDA hasher");
        let cpu_hasher = CpuHasher;

        let cuda_level = cuda_hasher.calculate_level(public_key, counter);
        let cpu_level = cpu_hasher.calculate_level(public_key, counter);

        assert_eq!(cuda_level, cpu_level,
            "CUDA and CPU implementations should produce the same result for long messages");
    }
}
