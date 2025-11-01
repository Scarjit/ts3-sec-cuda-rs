//! Security Level Improver for TeamSpeak 3 Identities
//!
//! This module provides functionality to improve the security level of a TS3 identity
//! by searching for counter values that produce SHA-1 hashes with more trailing zero bits.

use std::path::Path;
use std::time::{Duration, Instant};
use ini::Ini;
use crate::identity::{Ts3Identity, IdentityError};
use crate::helpers;

const BATCH_SIZE: u64 = 10_000; // Check this many counters before printing progress

/// Trait for implementing different hashing strategies (CPU, GPU, etc.)
pub trait SecurityLevelHasher {
    /// Calculate the security level for a given public key and counter
    fn calculate_level(&self, public_key: &str, counter: u64) -> u8;

    /// Get the name of this hasher (for display purposes)
    fn name(&self) -> &str;
}

/// Result of a level improvement search
#[derive(Debug, Clone)]
pub struct LevelSearchResult {
    pub counter: u64,
    pub level: u8,
}

/// Statistics about the improvement process
#[derive(Debug, Clone)]
pub struct ImprovementStatistics {
    pub hashes_checked: u64,
    pub elapsed_secs: f64,
}

/// Improve the security level of an identity by searching for better counter values
pub struct LevelImprover<H: SecurityLevelHasher> {
    identity: Ts3Identity,
    original_file_path: String,
    base_filename: String, // Filename without extension
    obfuscated_key: String, // The part after 'V' in the identity string
    hasher: H,
    best_level: u8,
    best_counter: u64,
    current_counter: u64,
    hashes_checked: u64,
    last_save: Instant,
    start_time: Instant,
}

impl<H: SecurityLevelHasher> LevelImprover<H> {
    /// Create a new LevelImprover for the given identity file
    pub fn new<P: AsRef<Path>>(path: P, hasher: H) -> Result<Self, IdentityError> {
        let path_str = path.as_ref().to_string_lossy().to_string();
        let identity = Ts3Identity::from_file(&path)?;

        // Extract the base filename (without extension)
        let base_filename = path.as_ref()
            .file_stem()
            .and_then(|s| s.to_str())
            .ok_or_else(|| IdentityError::IniError("Invalid filename".to_string()))?
            .to_string();

        // Extract the obfuscated key part from the INI file
        let conf = Ini::load_from_file(&path)
            .map_err(|e| IdentityError::IniError(e.to_string()))?;

        let identity_section = conf
            .section(Some("Identity"))
            .ok_or(IdentityError::MissingIdentitySection)?;

        let identity_value = identity_section
            .get("identity")
            .ok_or(IdentityError::MissingIdentityKey)?;

        // Parse the identity value to extract the obfuscated key
        // Format: "counter + 'V' + obfuscated_key"
        let identity_str = identity_value.trim().trim_matches('"');
        let v_pos = identity_str.find('V')
            .ok_or_else(|| IdentityError::InvalidFormat("No 'V' marker found".to_string()))?;
        let obfuscated_key = identity_str[v_pos + 1..].to_string();

        let current_level = identity.security_level();
        let current_counter = identity.counter;

        println!("Loaded identity from {}", path_str);
        println!("Current counter: {}", current_counter);
        println!("Current security level: {}", current_level);
        println!("Using hasher: {}", hasher.name());

        Ok(Self {
            identity,
            original_file_path: path_str,
            base_filename,
            obfuscated_key,
            hasher,
            best_level: current_level,
            best_counter: current_counter,
            current_counter: current_counter + 1,
            hashes_checked: 0,
            last_save: Instant::now(),
            start_time: Instant::now(),
        })
    }

    /// Calculate the security level for a given counter value
    fn calculate_level(&self, counter: u64) -> u8 {
        let omega = self.identity.public_key_base64();
        self.hasher.calculate_level(&omega, counter)
    }

    /// Get statistics about the improvement process
    pub fn get_statistics(&self) -> ImprovementStatistics {
        let elapsed = self.start_time.elapsed();
        let elapsed_secs = elapsed.as_secs_f64();

        ImprovementStatistics {
            hashes_checked: self.hashes_checked,
            elapsed_secs,
        }
    }

    /// Save the current best identity to a new file with format: basename-<level>.ini
    /// Automatically deletes intermediate level files, keeping only the base and latest
    fn save_progress(&mut self) -> Result<String, IdentityError> {
        use std::fs;

        // Create a new identity string with the updated counter
        // Format: "counter" + "V" + obfuscated_key
        let new_identity_value = format!("{}V{}", self.best_counter, self.obfuscated_key);

        // Read the original INI file to preserve structure
        let conf = Ini::load_from_file(&self.original_file_path)
            .map_err(|e| IdentityError::IniError(e.to_string()))?;

        let mut new_conf = Ini::new();

        for (section, props) in conf.iter() {
            if let Some(section_name) = section {
                for (key, value) in props.iter() {
                    if section_name == "Identity" && key == "identity" {
                        // Update with new counter + obfuscated key
                        new_conf.with_section(Some(section_name))
                            .set(key, format!("\"{}\"", new_identity_value));
                    } else {
                        new_conf.with_section(section).set(key, value);
                    }
                }
            }
        }

        // Create new filename: basename-<level>.ini
        let original_path = Path::new(&self.original_file_path);
        let parent_dir = original_path.parent()
            .ok_or_else(|| IdentityError::IniError("Invalid file path".to_string()))?;

        let new_filename = format!("{}-{}.ini", self.base_filename, self.best_level);
        let new_path = parent_dir.join(&new_filename);
        let new_path_str = new_path.to_string_lossy().to_string();

        // Delete intermediate files (files matching basename-*.ini but not the current level or original)
        if let Ok(entries) = fs::read_dir(parent_dir) {
            for entry in entries.filter_map(|e| e.ok()) {
                let path = entry.path();
                if let Some(filename) = path.file_name().and_then(|n| n.to_str()) {
                    // Check if this is an intermediate file we should delete
                    if filename.starts_with(&format!("{}-", self.base_filename))
                        && filename.ends_with(".ini")
                        && filename != new_filename.as_str() {
                        // Extract the level from the filename
                        if let Some(level_str) = filename
                            .strip_prefix(&format!("{}-", self.base_filename))
                            .and_then(|s| s.strip_suffix(".ini"))
                        {
                            if level_str.parse::<u8>().is_ok() {
                                // This is an intermediate level file, delete it
                                if let Err(e) = fs::remove_file(&path) {
                                    eprintln!("   Warning: Failed to delete intermediate file {}: {}",
                                              filename, e);
                                } else {
                                    println!("   🗑️  Deleted intermediate file: {}", filename);
                                }
                            }
                        }
                    }
                }
            }
        }

        // Write the new file
        new_conf.write_to_file(&new_path)
            .map_err(|e| IdentityError::IniError(e.to_string()))?;

        self.last_save = Instant::now();
        Ok(new_path_str)
    }

    /// Run the improvement search, calling the callback whenever a new level is found
    /// The callback should return `false` to stop the search, or `true` to continue
    pub fn improve<F>(&mut self, mut on_new_level: F) -> Result<(), IdentityError>
    where
        F: FnMut(&LevelSearchResult) -> bool,
    {
        println!("\nStarting level improvement search...");
        println!("Press Ctrl+C to stop\n");

        let mut last_print = Instant::now();
        let mut should_continue = true;

        loop {
            if !should_continue {
                return Ok(());
            }
            // Check a batch of counters
            for _ in 0..BATCH_SIZE {
                let level = self.calculate_level(self.current_counter);
                self.hashes_checked += 1;

                if level > self.best_level {
                    self.best_level = level;
                    self.best_counter = self.current_counter;

                    let result = LevelSearchResult {
                        counter: self.best_counter,
                        level: self.best_level,
                    };

                    println!("\n🎉 NEW LEVEL FOUND!");
                    println!("   Counter: {}", result.counter);
                    println!("   Level: {}", result.level);
                    println!("   Hashes checked: {}", self.hashes_checked);

                    // Call the callback and check if we should continue
                    should_continue = on_new_level(&result);

                    // Save immediately when we find a new level
                    let saved_path = self.save_progress()?;
                    println!("   ✓ Progress saved to {}\n", saved_path);

                    if !should_continue {
                        return Ok(());
                    }
                }

                self.current_counter += 1;
            }

            // Print progress every second
            if last_print.elapsed() >= Duration::from_secs(1) {
                helpers::print_progress(self.best_level, self.current_counter, self.hashes_checked, self.start_time);
                last_print = Instant::now();
            }

            // Note: We only save when finding new levels, not periodically
            // This prevents creating unnecessary files
        }
    }
}

