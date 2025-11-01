//! Helper utilities for the TS3 security level tool

use std::io::Write;
use std::time::Instant;

/// Format a number with thousand separators
pub fn format_number(n: u64) -> String {
    let s = n.to_string();
    let mut result = String::new();
    for (i, c) in s.chars().rev().enumerate() {
        if i > 0 && i % 3 == 0 {
            result.push(',');
        }
        result.push(c);
    }
    result.chars().rev().collect()
}

/// Print statistics about the hashing process
pub fn print_statistics(hashes_checked: u64, elapsed_secs: f64) {
    let avg_hashrate = if elapsed_secs > 0.0 {
        hashes_checked as f64 / elapsed_secs
    } else {
        0.0
    };

    println!("\n📊 Statistics:");
    println!("   Total hashes checked: {}", format_number(hashes_checked));
    println!("   Average hashrate: {:.2} MH/s", avg_hashrate / 1_000_000.0);
    if elapsed_secs < 1.0 {
        println!("   Time elapsed: {:.3}s", elapsed_secs);
    } else {
        println!("   Time elapsed: {:.1}s", elapsed_secs);
    }
}

/// Print progress line with current level, counter, hashrate, and total hashes
pub fn print_progress(best_level: u8, current_counter: u64, hashes_checked: u64, start_time: Instant) {
    let elapsed = start_time.elapsed();
    let hashes_per_sec = hashes_checked as f64 / elapsed.as_secs_f64();

    print!("\r[Level {}] Counter: {} | {:.0} H/s | {} hashes checked",
           best_level,
           current_counter,
           hashes_per_sec,
           format_number(hashes_checked));
    std::io::stdout().flush().ok();
}

/// Count trailing zero bits in a byte array (used for security level calculation)
pub fn count_trailing_zero_bits(hash: &[u8]) -> u8 {
    let mut count = 0;
    for &byte in hash {
        if byte == 0 {
            count += 8;
        } else {
            count += byte.trailing_zeros() as u8;
            break;
        }
    }
    count
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_format_number() {
        assert_eq!(format_number(1234567), "1,234,567");
        assert_eq!(format_number(42), "42");
        assert_eq!(format_number(1000), "1,000");
    }

    #[test]
    fn test_count_trailing_zero_bits() {
        assert_eq!(count_trailing_zero_bits(&[0, 0, 0, 0]), 32);
        assert_eq!(count_trailing_zero_bits(&[0x01]), 0);
        assert_eq!(count_trailing_zero_bits(&[0x02]), 1);
        assert_eq!(count_trailing_zero_bits(&[0x04]), 2);
        assert_eq!(count_trailing_zero_bits(&[0x00, 0x08]), 11);
    }
}
