//! CPU-based SHA-1 hasher implementation for TS3 security level calculation

use sha1::{Digest, Sha1};
use crate::level_improver::SecurityLevelHasher;
use crate::helpers::count_trailing_zero_bits;

/// CPU-based hasher using sha1 crate
pub struct CpuHasher;

impl SecurityLevelHasher for CpuHasher {
    fn calculate_level(&self, public_key: &str, counter: u64) -> u8 {
        let mut hasher = Sha1::new();
        hasher.update(public_key.as_bytes());
        hasher.update(counter.to_string().as_bytes());
        let hash = hasher.finalize();

        count_trailing_zero_bits(&hash)
    }

    fn name(&self) -> &str {
        "CPU"
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_calculate_level() {
        let hasher = CpuHasher;
        let public_key = "ME0DAgcAAgEgAiEAy/hhqSBja7A6FTZG5s+BMnQfCqYyS9sGsbyMKBb7spYCIQCBEtZWrZtewnxuh2hsigJswGHchu3XcaiQDZziMsxTsA==";

        // Known test case: counter 14 should produce level 8
        let level = hasher.calculate_level(public_key, 14);
        assert_eq!(level, 8);

        // Known test case: counter 201 should produce level 9
        let level = hasher.calculate_level(public_key, 201);
        assert_eq!(level, 9);

        // Known test case: counter 672 should produce level 12
        let level = hasher.calculate_level(public_key, 672);
        assert_eq!(level, 12);
    }

    #[test]
    fn test_hasher_name() {
        let hasher = CpuHasher;
        assert_eq!(hasher.name(), "CPU");
    }
}
