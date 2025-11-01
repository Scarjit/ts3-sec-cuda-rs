//! CUDA-based SHA-1 hasher implementation for TS3 security level calculation
//!
//! This module provides GPU-accelerated hashing using NVIDIA CUDA.
//! Currently not implemented - returns errors.

use crate::level_improver::SecurityLevelHasher;

/// CUDA-based hasher using GPU acceleration
#[derive(Debug)]
pub struct CudaHasher;

impl CudaHasher {
    /// Create a new CUDA hasher
    ///
    /// # Errors
    ///
    /// Returns an error if CUDA is not available or not yet implemented
    pub fn new() -> Result<Self, CudaError> {
        Err(CudaError::NotImplemented)
    }
}

impl SecurityLevelHasher for CudaHasher {
    fn calculate_level(&self, _public_key: &str, _counter: u64) -> u8 {
        // This should never be called since new() returns an error
        panic!("CUDA hasher not yet implemented");
    }

    fn name(&self) -> &str {
        "CUDA"
    }
}

/// Errors that can occur with CUDA operations
#[derive(Debug, thiserror::Error)]
pub enum CudaError {
    #[error("CUDA support not yet implemented")]
    NotImplemented,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_cuda_not_implemented() {
        let result = CudaHasher::new();
        assert!(result.is_err());
        assert!(matches!(result.unwrap_err(), CudaError::NotImplemented));
    }

    #[test]
    fn test_hasher_name() {
        // Even though we can't construct it normally, we can test the name
        let hasher = CudaHasher;
        assert_eq!(hasher.name(), "CUDA");
    }
}
