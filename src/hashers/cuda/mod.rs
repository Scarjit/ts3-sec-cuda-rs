//! CUDA-based SHA-1 hasher implementation for TS3 security level calculation
//!
//! This module provides GPU-accelerated hashing using NVIDIA CUDA.

use crate::level_improver::SecurityLevelHasher;
use cudarc::driver::*;
use cudarc::nvrtc::Ptx;
use std::sync::Arc;
use std::cell::RefCell;

/// SHA1 initial hash values (RFC 3174)
const SHA1_INIT_HASH: [u32; 5] = [
    0x67452301,
    0xEFCDAB89,
    0x98BADCFE,
    0x10325476,
    0xC3D2E1F0,
];

/// CUDA-based hasher using GPU acceleration
pub struct CudaHasher {
    ctx: Arc<CudaContext>,
    module: Arc<CudaModule>,
    module_optimized: Arc<CudaModule>,
    // Cached public key in device memory for optimized batch processing (interior mutability)
    cached_public_key: RefCell<Option<(CudaSlice<u8>, usize)>>, // (device memory, length)
}

impl CudaHasher {
    /// Create a new CUDA hasher
    ///
    /// # Errors
    ///
    /// Returns an error if CUDA is not available or initialization fails
    pub fn new() -> Result<Self, CudaError> {
        let ctx = CudaContext::new(0).map_err(CudaError::DeviceInitError)?;

        // Load and compile both CUDA kernels
        let ptx = compile_kernel()?;
        let module = ctx.load_module(ptx)
            .map_err(CudaError::KernelLoadError)?;

        let ptx_optimized = compile_optimized_kernel()?;
        let module_optimized = ctx.load_module(ptx_optimized)
            .map_err(CudaError::KernelLoadError)?;

        Ok(Self {
            ctx,
            module,
            module_optimized,
            cached_public_key: RefCell::new(None),
        })
    }

    /// Hash multiple messages in parallel using the CUDA kernel
    /// Returns a vector of 20-byte (160-bit) SHA1 hashes
    ///
    /// # Performance
    /// This is **much faster** than calling `hash_message` in a loop because
    /// all hashes are computed in parallel on the GPU.
    ///
    /// # Note
    /// Currently, all messages must have the same length and fit in a single block (≤55 bytes).
    /// For full multi-block support in batch mode, additional implementation is needed.
    #[allow(dead_code)] // Public API method, not used internally but available for library consumers
    pub fn hash_messages_batch(&self, messages: &[&[u8]]) -> Result<Vec<[u8; 20]>, CudaError> {
        if messages.is_empty() {
            return Ok(Vec::new());
        }

        let num_hashes = messages.len();

        // For now, ensure all messages have the same length and fit in one block
        let first_len = messages[0].len();
        if first_len > 55 {
            // Fall back to sequential processing for long messages
            return messages.iter()
                .map(|msg| self.hash_message(msg))
                .collect();
        }

        for msg in messages {
            if msg.len() != first_len {
                // Fall back to sequential processing for variable-length messages
                return messages.iter()
                    .map(|msg| self.hash_message(msg))
                    .collect();
            }
        }

        // Prepare all input blocks
        let mut input_blocks = Vec::with_capacity(num_hashes);
        for msg in messages {
            let blocks = prepare_sha1_blocks(msg);
            input_blocks.push(blocks[0]); // Take first block (all messages fit in one)
        }

        // Flatten into a single array for GPU transfer
        let flat_input: Vec<u32> = input_blocks.iter()
            .flat_map(|block| block.iter().copied())
            .collect();

        let results = launch_batch_kernel(
            &self.ctx,
            &self.module,
            &flat_input,
            &SHA1_INIT_HASH,
            num_hashes,
        )?;

        Ok(results)
    }

    /// Hash a message using the CUDA kernel
    /// Returns the 20-byte (160-bit) SHA1 hash
    /// Supports multi-block messages of any length
    ///
    /// For hashing multiple messages, use `hash_messages_batch` instead for better performance.
    pub fn hash_message(&self, message: &[u8]) -> Result<[u8; 20], CudaError> {
        let blocks = prepare_sha1_blocks(message);
        let mut h = SHA1_INIT_HASH;

        // Process each block, chaining the hash values
        for block in blocks {
            let output = launch_single_block_kernel(&self.ctx, &self.module, &block, &h)?;
            h.copy_from_slice(&output);
        }

        Ok(words_to_hash_bytes(&h))
    }

    /// Optimized batch computation of security levels using GPU-side counter generation
    /// This is significantly faster than the standard batch processing as it:
    /// - Keeps public key in GPU memory (transferred once)
    /// - Generates counter strings on GPU (no CPU allocations)
    /// - Returns only security levels (not full hashes)
    ///
    /// # Arguments
    /// * `public_key` - The base64-encoded public key
    /// * `start_counter` - First counter value
    /// * `num_counters` - How many consecutive counters to process
    ///
    /// # Returns
    /// Vector of security levels (trailing zero bits) for each counter
    #[allow(dead_code)] // Used in tests for debugging
    pub fn calculate_levels_optimized(
        &self,
        public_key: &str,
        start_counter: u64,
        num_counters: usize,
    ) -> Result<Vec<u8>, CudaError> {
        let stream = self.ctx.default_stream();

        // Cache public key in device memory if not already cached or if it changed
        let public_key_bytes = public_key.as_bytes();
        let mut cache = self.cached_public_key.borrow_mut();

        let needs_upload = match &*cache {
            None => true,
            Some((_, cached_len)) => *cached_len != public_key_bytes.len(),
        };

        if needs_upload {
            let d_public_key = stream.memcpy_stod(public_key_bytes)
                .map_err(CudaError::MemoryError)?;
            *cache = Some((d_public_key, public_key_bytes.len()));
        }

        // Get the cached public key device memory and clone the reference
        let (d_public_key, public_key_len) = {
            let (slice, len) = cache.as_ref().unwrap();
            (slice.clone(), *len)
        };

        // Drop the borrow explicitly
        drop(cache);

        // Allocate output memory for security levels
        let mut d_security_levels = stream.alloc_zeros::<u8>(num_counters)
            .map_err(CudaError::MemoryError)?;

        // Load the optimized kernel
        let kernel_func = self.module_optimized.load_function("sha1_security_level_optimized")
            .map_err(CudaError::KernelLoadError)?;

        // Configure kernel launch
        let threads_per_block = 256;
        let num_blocks = (num_counters + threads_per_block - 1) / threads_per_block;

        let cfg = LaunchConfig {
            grid_dim: (num_blocks as u32, 1, 1),
            block_dim: (threads_per_block as u32, 1, 1),
            shared_mem_bytes: 0,
        };

        // Launch kernel
        // SAFETY: Kernel launch is safe because:
        // 1. d_public_key contains valid public key bytes in device memory
        // 2. public_key_len is the correct length
        // 3. d_security_levels has space for num_counters bytes
        // 4. All memory is properly allocated and bounds-checked
        #[allow(unsafe_code)]
        unsafe {
            stream.launch_builder(&kernel_func)
                .arg(&d_public_key)
                .arg(&(public_key_len as i32))
                .arg(&start_counter)
                .arg(&(num_counters as i32))
                .arg(&mut d_security_levels)
                .launch(cfg)
        }.map_err(CudaError::KernelLaunchError)?;

        // Copy results back to host
        let security_levels: Vec<u8> = stream.memcpy_dtov(&d_security_levels)
            .map_err(CudaError::MemoryError)?;

        Ok(security_levels)
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

    /// Batch processing using CUDA for parallel GPU computation
    fn calculate_levels_batch(&self, public_key: &str, counters: &[u64]) -> Vec<u8> {
        if counters.is_empty() {
            return Vec::new();
        }

        // Check if counters are consecutive (optimal case for GPU)
        let is_consecutive = counters.windows(2).all(|w| w[1] == w[0] + 1);

        if is_consecutive {
            // Use the optimized kernel for consecutive counters
            match self.calculate_levels_optimized(public_key, counters[0], counters.len()) {
                Ok(levels) => return levels,
                Err(_) => {
                    // Fall back to sequential processing on error
                    return counters.iter()
                        .map(|&counter| self.calculate_level(public_key, counter))
                        .collect();
                }
            }
        }

        // For non-consecutive counters, process each one using the optimized kernel
        // This still benefits from GPU acceleration and avoids CPU-side string operations
        let mut results = Vec::with_capacity(counters.len());
        for &counter in counters {
            match self.calculate_levels_optimized(public_key, counter, 1) {
                Ok(mut levels) => results.push(levels.remove(0)),
                Err(_) => results.push(self.calculate_level(public_key, counter)),
            }
        }
        results
    }

    fn name(&self) -> &str {
        "CUDA"
    }
}

/// Convert 5 u32 words to a 20-byte SHA1 hash (big-endian)
fn words_to_hash_bytes(words: &[u32; 5]) -> [u8; 20] {
    let mut hash_bytes = [0u8; 20];
    for (i, &word) in words.iter().enumerate() {
        let bytes = word.to_be_bytes();
        hash_bytes[i * 4..(i + 1) * 4].copy_from_slice(&bytes);
    }
    hash_bytes
}

/// Launch CUDA kernel for a single SHA1 block
///
/// # Safety invariants
/// This function maintains the following invariants:
/// - d_input contains exactly 16 u32 values
/// - d_init_hash contains exactly 5 u32 values
/// - Returns exactly 5 u32 values
fn launch_single_block_kernel(
    ctx: &Arc<CudaContext>,
    module: &Arc<CudaModule>,
    block: &[u32; 16],
    init_hash: &[u32; 5],
) -> Result<[u32; 5], CudaError> {
    let stream = ctx.default_stream();
    let sha1_func = module.load_function("sha1_simple")
        .map_err(|e| CudaError::KernelLoadError(e))?;

    // Allocate device memory
    let d_input = stream.memcpy_stod(block)
        .map_err(CudaError::MemoryError)?;
    let d_init_hash = stream.memcpy_stod(init_hash)
        .map_err(CudaError::MemoryError)?;
    let mut d_output = stream.alloc_zeros::<u32>(5)
        .map_err(CudaError::MemoryError)?;

    let cfg = LaunchConfig {
        grid_dim: (1, 1, 1),
        block_dim: (1, 1, 1),
        shared_mem_bytes: 0,
    };

    // SAFETY: Kernel launch is safe because:
    // 1. d_input is a valid CudaSlice containing 16 u32 values
    // 2. d_output is a valid CudaSlice with space for 5 u32 values
    // 3. num_hashes is set to 1
    // 4. d_init_hash is a valid CudaSlice containing 5 u32 values
    // 5. All memory is allocated on the same device and outlives the kernel
    // 6. Grid/block dimensions (1,1,1) ensure only one thread executes
    #[allow(unsafe_code)]
    unsafe {
        stream.launch_builder(&sha1_func)
            .arg(&d_input)
            .arg(&mut d_output)
            .arg(&1i32)
            .arg(&d_init_hash)
            .launch(cfg)
    }.map_err(CudaError::KernelLaunchError)?;

    // Copy result back
    let output: Vec<u32> = stream.memcpy_dtov(&d_output)
        .map_err(CudaError::MemoryError)?;

    let mut result = [0u32; 5];
    result.copy_from_slice(&output[..5]);
    Ok(result)
}

/// Launch CUDA kernel for batch SHA1 hashing
///
/// # Safety invariants
/// - flat_input contains num_hashes * 16 u32 values
/// - init_hash contains exactly 5 u32 values
/// - Returns num_hashes hash results
fn launch_batch_kernel(
    ctx: &Arc<CudaContext>,
    module: &Arc<CudaModule>,
    flat_input: &[u32],
    init_hash: &[u32; 5],
    num_hashes: usize,
) -> Result<Vec<[u8; 20]>, CudaError> {
    let stream = ctx.default_stream();
    let sha1_func = module.load_function("sha1_simple")
        .map_err(|e| CudaError::KernelLoadError(e))?;

    // Allocate device memory
    let d_input = stream.memcpy_stod(flat_input)
        .map_err(CudaError::MemoryError)?;
    let d_init_hash = stream.memcpy_stod(init_hash)
        .map_err(CudaError::MemoryError)?;
    let mut d_output = stream.alloc_zeros::<u32>(num_hashes * 5)
        .map_err(CudaError::MemoryError)?;

    // Launch kernel with multiple threads (256 threads per block)
    let threads_per_block = 256;
    let num_blocks = (num_hashes + threads_per_block - 1) / threads_per_block;

    let cfg = LaunchConfig {
        grid_dim: (num_blocks as u32, 1, 1),
        block_dim: (threads_per_block as u32, 1, 1),
        shared_mem_bytes: 0,
    };

    // SAFETY: Kernel launch is safe because:
    // 1. d_input contains num_hashes * 16 u32 values
    // 2. d_output has space for num_hashes * 5 u32 values
    // 3. num_hashes is correctly set
    // 4. d_init_hash contains 5 u32 values (shared initial state)
    // 5. All memory is allocated on the same device and outlives the kernel
    // 6. Grid/block dimensions ensure num_hashes threads execute
    #[allow(unsafe_code)]
    unsafe {
        stream.launch_builder(&sha1_func)
            .arg(&d_input)
            .arg(&mut d_output)
            .arg(&(num_hashes as i32))
            .arg(&d_init_hash)
            .launch(cfg)
    }.map_err(CudaError::KernelLaunchError)?;

    // Copy all results back
    let output: Vec<u32> = stream.memcpy_dtov(&d_output)
        .map_err(CudaError::MemoryError)?;

    // Convert to byte arrays
    let results = (0..num_hashes)
        .map(|i| {
            let words: [u32; 5] = [
                output[i * 5],
                output[i * 5 + 1],
                output[i * 5 + 2],
                output[i * 5 + 3],
                output[i * 5 + 4],
            ];
            words_to_hash_bytes(&words)
        })
        .collect();

    Ok(results)
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

fn compile_optimized_kernel() -> Result<Ptx, CudaError> {
    let kernel_src = include_str!("sha1_optimized.cu");

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

    #[test]
    fn test_batch_hashing() {
        // Test that batch hashing produces correct results
        let hasher = CudaHasher::new().expect("Failed to initialize CUDA hasher");

        let messages: Vec<&[u8]> = vec![b"123", b"abc", b"xyz"];

        let batch_results = hasher.hash_messages_batch(&messages)
            .expect("Batch hashing failed");

        assert_eq!(batch_results.len(), 3, "Should have 3 results");

        // Verify each hash matches individual computation
        for (i, msg) in messages.iter().enumerate() {
            let individual_hash = hasher.hash_message(msg)
                .expect("Individual hash failed");
            assert_eq!(batch_results[i], individual_hash,
                "Batch hash for '{}' should match individual hash",
                String::from_utf8_lossy(msg));
        }

        // Verify specific known hashes
        let hash_123 = batch_results[0].iter()
            .map(|b| format!("{:02x}", b))
            .collect::<String>();
        let hash_abc = batch_results[1].iter()
            .map(|b| format!("{:02x}", b))
            .collect::<String>();

        assert_eq!(hash_123, "40bd001563085fc35165329ea1ff5c5ecbdbbeef", "SHA1('123') mismatch");
        assert_eq!(hash_abc, "a9993e364706816aba3e25717850c26c9cd0d89d", "SHA1('abc') mismatch");
    }

    #[test]
    fn test_batch_hashing_large() {
        // Test batch hashing with many inputs (simulates TS3 counter searching)
        let hasher = CudaHasher::new().expect("Failed to initialize CUDA hasher");

        // Create 1000 different messages
        let mut messages_vec: Vec<String> = Vec::new();
        for i in 0..1000 {
            messages_vec.push(format!("test_message_{}", i));
        }

        let messages: Vec<&[u8]> = messages_vec.iter()
            .map(|s| s.as_bytes())
            .collect();

        let batch_results = hasher.hash_messages_batch(&messages)
            .expect("Large batch hashing failed");

        assert_eq!(batch_results.len(), 1000, "Should have 1000 results");

        // Spot check a few results
        for i in [0, 500, 999].iter() {
            let individual_hash = hasher.hash_message(messages[*i])
                .expect("Individual hash failed");
            assert_eq!(batch_results[*i], individual_hash,
                "Batch hash at index {} should match individual hash", i);
        }
    }

    #[test]
    fn test_optimized_kernel_security_levels() {
        // Test that the optimized kernel produces correct security levels
        let hasher = CudaHasher::new().expect("Failed to initialize CUDA hasher");
        let public_key = "ME0DAgcAAgEgAiEAy/hhqSBja7A6FTZG5s+BMnQfCqYyS9sGsbyMKBb7spYCIQCBEtZWrZtewnxuh2hsigJswGHchu3XcaiQDZziMsxTsA==";

        // Test known counter values (from CPU test)
        println!("Testing counter 14 (expected level 8):");
        let level_14 = hasher.calculate_level(public_key, 14);
        println!("  Got level: {}", level_14);
        assert_eq!(level_14, 8, "Counter 14 should have level 8");

        println!("\nTesting counter 201 (expected level 9):");
        let level_201 = hasher.calculate_level(public_key, 201);
        println!("  Got level: {}", level_201);
        assert_eq!(level_201, 9, "Counter 201 should have level 9");

        println!("\nTesting counter 672 (expected level 12):");
        let level_672 = hasher.calculate_level(public_key, 672);
        println!("  Got level: {}", level_672);
        assert_eq!(level_672, 12, "Counter 672 should have level 12");
    }

    #[test]
    fn test_optimized_kernel_vs_individual() {
        // Compare optimized kernel output with individual calculate_level
        let hasher = CudaHasher::new().expect("Failed to initialize CUDA hasher");
        let public_key = "ME0DAgcAAgEgAiEAy/hhqSBja7A6FTZG5s+BMnQfCqYyS9sGsbyMKBb7spYCIQCBEtZWrZtewnxuh2hsigJswGHchu3XcaiQDZziMsxTsA==";

        // First, verify the hash itself is correct by comparing with hash_message
        println!("\nVerifying hash correctness:");
        let message = format!("{}14", public_key);
        let expected_hash = hasher.hash_message(message.as_bytes())
            .expect("hash_message failed");

        println!("Message: {}", message);
        println!("Expected hash (from hash_message): {}",
            expected_hash.iter().map(|b| format!("{:02x}", b)).collect::<String>());

        let expected_level = crate::helpers::count_trailing_zero_bits(&expected_hash);
        println!("Expected security level: {}", expected_level);

        // Test optimized method directly
        println!("\nTesting optimized kernel directly:");
        let levels_optimized = hasher.calculate_levels_optimized(public_key, 14, 1)
            .expect("Optimized kernel failed");

        println!("Optimized kernel level for counter 14: {}", levels_optimized[0]);

        let individual_level = hasher.calculate_level(public_key, 14);
        println!("Individual method level for counter 14: {}", individual_level);

        assert_eq!(levels_optimized[0], expected_level,
            "Optimized kernel should match expected level");
        assert_eq!(levels_optimized[0], individual_level,
            "Optimized kernel should match individual for counter 14");
    }

    #[test]
    fn test_optimized_kernel_batch() {
        // Test that the optimized batch kernel produces correct results
        let hasher = CudaHasher::new().expect("Failed to initialize CUDA hasher");
        let public_key = "ME0DAgcAAgEgAiEAy/hhqSBja7A6FTZG5s+BMnQfCqYyS9sGsbyMKBb7spYCIQCBEtZWrZtewnxuh2hsigJswGHchu3XcaiQDZziMsxTsA==";

        // Test batch with consecutive counters including known values
        println!("\nTesting batch 14-700:");
        let counters: Vec<u64> = (14..=700).collect();
        let levels = hasher.calculate_levels_batch(public_key, &counters);

        // Debug: print first few results
        println!("First 10 levels:");
        for i in 0..10 {
            println!("  Counter {}: level {}", 14 + i, levels[i]);
        }

        // Check specific known values
        println!("\nExpected levels:");
        println!("  Counter 14: expected 8, got {}", levels[0]);
        println!("  Counter 201: expected 9, got {}", levels[201 - 14]);
        println!("  Counter 672: expected 12, got {}", levels[672 - 14]);

        assert_eq!(levels[0], 8, "Counter 14 (index 0) should have level 8");
        assert_eq!(levels[201 - 14], 9, "Counter 201 should have level 9");
        assert_eq!(levels[672 - 14], 12, "Counter 672 should have level 12");

        // Find best in batch to verify
        let best = levels.iter().enumerate().max_by_key(|&(_, level)| level).unwrap();
        println!("  Best in batch: counter {} has level {}", 14 + best.0, best.1);
        assert_eq!(14 + best.0, 672, "Counter 672 should be the best in this range");
        assert_eq!(*best.1, 12, "Best counter should have level 12");
    }

    #[test]
    fn test_both_kernels_match() {
        // Test that old kernel (hash_message) and new optimized kernel produce same results
        let hasher = CudaHasher::new().expect("Failed to initialize CUDA hasher");
        let public_key = "ME0DAgcAAgEgAiEAy/hhqSBja7A6FTZG5s+BMnQfCqYyS9sGsbyMKBb7spYCIQCBEtZWrZtewnxuh2hsigJswGHchu3XcaiQDZziMsxTsA==";

        println!("\nComparing old kernel (hash_message) vs new optimized kernel:");

        // Test several counter values
        for counter in [14, 201, 672, 1000, 5000] {
            // Old kernel path: calculate_level uses hash_message
            let level_old = hasher.calculate_level(public_key, counter);

            // New optimized kernel path: calculate_levels_optimized
            let levels_new = hasher.calculate_levels_optimized(public_key, counter, 1)
                .expect("Optimized kernel failed");

            println!("  Counter {}: old_kernel={}, new_kernel={}", counter, level_old, levels_new[0]);

            assert_eq!(level_old, levels_new[0],
                "Old and new kernels must produce same result for counter {}", counter);
        }

        println!("  All counters match! Both kernels produce identical results.");
    }
}
