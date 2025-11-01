//! TeamSpeak 3 Identity Parser
//!
//! This module provides functionality to parse TeamSpeak 3 identity.ini files.
//!
//! The identity format is: `COUNTER || 'V' || base64encode(obfuscate(base64encode(KEYPAIR_ASN1)))`
//! where:
//! - `COUNTER`: A numeric counter (u64) indicating the identity security level
//! - `'V'`: A version marker character
//! - The private key is base64-encoded, obfuscated, and base64-encoded again

use std::path::Path;
use thiserror::Error;
use ini::Ini;
use tsproto_types::crypto::EccKeyPrivP256;
use sha1::{Digest, Sha1};

#[derive(Debug, Error)]
pub enum IdentityError {
    #[error("Failed to read identity file: {0}")]
    IoError(#[from] std::io::Error),

    #[error("Failed to parse INI file: {0}")]
    IniError(String),

    #[error("Missing [Identity] section in file")]
    MissingIdentitySection,

    #[error("Missing 'identity' key in [Identity] section")]
    MissingIdentityKey,

    #[error("Invalid identity format: {0}")]
    InvalidFormat(String),

    #[error("Failed to parse counter: {0}")]
    InvalidCounter(#[from] std::num::ParseIntError),

    #[error("Failed to parse private key: {0}")]
    PrivateKeyError(String),
}

#[derive(Debug, Clone)]
pub struct Ts3Identity {
    pub counter: u64,
    pub private_key: EccKeyPrivP256,
}

impl Ts3Identity {
    /// Parse a TeamSpeak 3 identity.ini file
    pub fn from_file<P: AsRef<Path>>(path: P) -> Result<Self, IdentityError> {
        let conf = Ini::load_from_file(path)
            .map_err(|e| IdentityError::IniError(e.to_string()))?;

        Self::from_ini(conf)
    }

    /// Parse a TeamSpeak 3 identity from INI string content
    pub fn from_ini_str(ini_content: &str) -> Result<Self, IdentityError> {
        let conf = Ini::load_from_str(ini_content)
            .map_err(|e| IdentityError::IniError(e.to_string()))?;

        Self::from_ini(conf)
    }

    /// Parse identity from an Ini object
    fn from_ini(conf: Ini) -> Result<Self, IdentityError> {
        let identity_section = conf
            .section(Some("Identity"))
            .ok_or(IdentityError::MissingIdentitySection)?;

        let identity_value = identity_section
            .get("identity")
            .ok_or(IdentityError::MissingIdentityKey)?;

        Self::parse_identity(identity_value)
    }

    /// Parse an identity string in the format "counter" followed by base64-encoded private key
    /// Example: "1118470V0W/upAdIHOOe56XP5QOtgKfSBs9b..."
    pub fn parse_identity(identity_str: &str) -> Result<Self, IdentityError> {
        // Remove surrounding quotes if present
        let identity_str = identity_str.trim().trim_matches('"');

        // Find where the counter ends (first non-digit character)
        let counter_end = identity_str
            .find(|c: char| !c.is_ascii_digit())
            .ok_or_else(|| IdentityError::InvalidFormat("No private key found".to_string()))?;

        if counter_end == 0 {
            return Err(IdentityError::InvalidFormat("No counter found".to_string()));
        }

        let counter = identity_str[..counter_end].parse::<u64>()?;
        let key_str = &identity_str[counter_end..];

        // The format is: COUNTER || 'V' || base64encode(obfuscate(base64encode(KEYPAIR_ASN1)))
        // Skip the 'V' prefix
        if !key_str.starts_with('V') {
            return Err(IdentityError::InvalidFormat(
                format!("Expected 'V' prefix after counter, found: {}",
                    key_str.chars().next().unwrap_or(' '))
            ));
        }
        let obfuscated_key = &key_str[1..]; // Skip the 'V'

        // Parse the obfuscated private key using tsproto-types
        let private_key = EccKeyPrivP256::from_ts_obfuscated(obfuscated_key)
            .map_err(|e| IdentityError::PrivateKeyError(format!("{:?}", e)))?;

        Ok(Ts3Identity {
            counter,
            private_key,
        })
    }

    /// Get the private key as a base64-encoded TS string
    pub fn private_key_base64(&self) -> String {
        self.private_key.to_ts()
    }

    /// Get the public key (omega) as a base64-encoded TS string
    pub fn public_key_base64(&self) -> String {
        self.private_key.to_pub().to_ts()
    }

    /// Calculate the security level for this identity
    ///
    /// The security level is the number of trailing zero bits in SHA-1(public_key || counter)
    pub fn security_level(&self) -> u8 {
        let omega = self.public_key_base64();

        let mut hasher = Sha1::new();
        hasher.update(omega.as_bytes());
        hasher.update(self.counter.to_string().as_bytes());
        let hash = hasher.finalize();

        count_trailing_zero_bits(&hash)
    }
}

/// Count trailing zero bits in a byte array
fn count_trailing_zero_bits(hash: &[u8]) -> u8 {
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
    fn test_parse_identity_invalid_format_no_counter() {
        let identity_str = "just_text_no_counter";
        let result = Ts3Identity::parse_identity(identity_str);

        assert!(matches!(result, Err(IdentityError::InvalidFormat(_))));
    }

    #[test]
    fn test_parse_identity_only_counter() {
        let identity_str = "12345";
        let result = Ts3Identity::parse_identity(identity_str);

        assert!(matches!(result, Err(IdentityError::InvalidFormat(_))));
    }

    #[test]
    fn test_parse_identity_invalid_key() {
        let identity_str = "42VInvalidBase64!!!";
        let result = Ts3Identity::parse_identity(identity_str);

        assert!(matches!(result, Err(IdentityError::PrivateKeyError(_))));
    }

    #[test]
    fn test_parse_identity_missing_v_prefix() {
        let identity_str = "42XSomeKey";
        let result = Ts3Identity::parse_identity(identity_str);

        assert!(matches!(result, Err(IdentityError::InvalidFormat(_))));
    }

    #[test]
    fn test_parse_full_config_file(){
        let config = "
        [Identity]
id=DO NOT USE IN PRODUCTION - TEST KEY ONLY !
identity=\"1118470V0W/upAdIHOOe56XP5QOtgKfSBs9bAXwCVVUYRzFEBE0gHFcFXkJODQJ1LwBXUl8IDGNLB1dzfjM9ZH0GVFlFY0MeaFZNE04FGBUeUiM1EAd+BF1ZXF4YWlAwBlJkCH8DU3NCdUhLOENJUUNQd3ZWYzVzOUx3aGxaSk0yTUk0djUzSkw5Ykp5ekoyVU0wNFVWZkNQUUlRPT0=\"
nickname=TeamSpeakUser
phonetic_nickname=
";
        let result = Ts3Identity::from_ini_str(config);
        assert!(result.is_ok(), "Result should be ok, is: {:?}", result);

        let identity = result.unwrap();
        assert_eq!(identity.counter, 1118470);
    }

    #[test]
    fn test_security_level() {
        let config = "[Identity]\nidentity=\"1118470V0W/upAdIHOOe56XP5QOtgKfSBs9bAXwCVVUYRzFEBE0gHFcFXkJODQJ1LwBXUl8IDGNLB1dzfjM9ZH0GVFlFY0MeaFZNE04FGBUeUiM1EAd+BF1ZXF4YWlAwBlJkCH8DU3NCdUhLOENJUUNQd3ZWYzVzOUx3aGxaSk0yTUk0djUzSkw5Ykp5ekoyVU0wNFVWZkNQUUlRPT0=\"";
        let identity = Ts3Identity::from_ini_str(config).unwrap();

        let level = identity.security_level();
        println!("Security level for counter {}: {}", identity.counter, level);

        // The security level should be at least 16 (typically identities aim for level 20+)
        assert!(level >= 16, "Security level should be at least 16, got {}", level);
    }

    #[test]
    fn test_count_trailing_zero_bits() {
        // Test with all zeros
        assert_eq!(count_trailing_zero_bits(&[0, 0, 0, 0]), 32);

        // Test with one byte = 0x01 (binary: 00000001) -> 0 trailing zeros
        assert_eq!(count_trailing_zero_bits(&[0x01]), 0);

        // Test with one byte = 0x02 (binary: 00000010) -> 1 trailing zero
        assert_eq!(count_trailing_zero_bits(&[0x02]), 1);

        // Test with one byte = 0x04 (binary: 00000100) -> 2 trailing zeros
        assert_eq!(count_trailing_zero_bits(&[0x04]), 2);

        // Test with first byte zero, second byte = 0x08 (binary: 00001000) -> 8 + 3 = 11 trailing zeros
        assert_eq!(count_trailing_zero_bits(&[0x00, 0x08]), 11);
    }
}
