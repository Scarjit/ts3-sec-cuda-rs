#[deny(unsafe_code,unused)]
mod identity;
mod level_improver;
mod hashers;
mod helpers;
mod cli;

use identity::Ts3Identity;
use level_improver::{LevelImprover, SecurityLevelHasher};
use hashers::{CpuHasher, CudaHasher};
use helpers::{print_statistics, format_number};
use cli::{Cli, Command, HasherMethod};
use clap::Parser;
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::Arc;

fn main() {
    let cli = Cli::parse();

    match cli.command {
        Command::Decode { file, string } => {
            decode_identity(file, string);
        }
        Command::Increase { file, string, target, method } => {
            increase_level(file, string, target, method);
        }
    }
}

fn decode_identity(file: Option<String>, string: Option<String>) {
    let identity = if let Some(file_path) = file {
        match Ts3Identity::from_file(&file_path) {
            Ok(id) => {
                println!("📄 Loaded from file: {}\n", file_path);
                id
            }
            Err(e) => {
                eprintln!("❌ Error loading file '{}': {}", file_path, e);
                std::process::exit(1);
            }
        }
    } else if let Some(identity_str) = string {
        match Ts3Identity::parse_identity(&identity_str) {
            Ok(id) => {
                println!("📝 Parsed from string\n");
                id
            }
            Err(e) => {
                eprintln!("❌ Error parsing identity string: {}", e);
                std::process::exit(1);
            }
        }
    } else {
        eprintln!("❌ Error: Must provide either --file or --string");
        std::process::exit(1);
    };

    // Display identity information
    println!("Identity Information:");
    println!("  Counter:        {}", identity.counter);
    println!("  Security Level: {}", identity.security_level());
    println!("  Public Key:     {}", identity.public_key_base64());
    println!("\n💡 Proof of Work:");
    println!("  This identity proves ~{:.0} SHA-1 hashes were computed.",
             2_f64.powi(identity.security_level() as i32));
}

fn increase_level(file: Option<String>, string: Option<String>, target_level: u8, method: HasherMethod) {
    println!("🚀 TeamSpeak 3 Security Level Improver\n");

    // Determine input mode
    let (current_identity, input_source) = if let Some(file_path) = &file {
        match Ts3Identity::from_file(file_path) {
            Ok(id) => (id, format!("File: {}", file_path)),
            Err(e) => {
                eprintln!("❌ Error loading file '{}': {}", file_path, e);
                std::process::exit(1);
            }
        }
    } else if let Some(identity_str) = &string {
        match Ts3Identity::parse_identity(identity_str) {
            Ok(id) => (id, "Identity string".to_string()),
            Err(e) => {
                eprintln!("❌ Error parsing identity string: {}", e);
                std::process::exit(1);
            }
        }
    } else {
        eprintln!("❌ Error: Must provide either --file or --string");
        std::process::exit(1);
    };

    let current_level = current_identity.security_level();
    println!("📄 {}", input_source);
    println!("📊 Current level: {}", current_level);
    println!("🎯 Target level:  {}", target_level);

    if current_level >= target_level {
        println!("\n✅ Identity already at or above target level!");
        return;
    }

    // Setup Ctrl+C handler
    ctrlc::set_handler(move || {
        println!("\n\n⚠️  Ctrl+C received, stopping gracefully...");
        std::process::exit(0);
    }).expect("Error setting Ctrl-C handler");

    // Create improver based on method
    match method {
        HasherMethod::Cpu => {
            println!("⚙️  Method: CPU\n");
            if let Some(file_path) = file {
                run_improver_file(&file_path, target_level, CpuHasher);
            } else if let Some(identity_str) = string {
                run_improver_string(&identity_str, target_level, CpuHasher);
            }
        }
        HasherMethod::Cuda => {
            println!("⚙️  Method: CUDA\n");
            match CudaHasher::new() {
                Ok(cuda_hasher) => {
                    if let Some(file_path) = file {
                        run_improver_file(&file_path, target_level, cuda_hasher);
                    } else if let Some(identity_str) = string {
                        run_improver_string(&identity_str, target_level, cuda_hasher);
                    }
                }
                Err(e) => {
                    eprintln!("❌ CUDA Error: {}", e);
                    eprintln!("   CUDA support is not yet implemented.");
                    eprintln!("   Please use --method cpu for now.");
                    std::process::exit(1);
                }
            }
        }
    }
}

fn run_improver_file<H: SecurityLevelHasher>(
    file_path: &str,
    target_level: u8,
    hasher: H,
) {
    let mut improver = match LevelImprover::new(file_path, hasher) {
        Ok(imp) => imp,
        Err(e) => {
            eprintln!("❌ Error initializing level improver: {}", e);
            std::process::exit(1);
        }
    };

    // Flag to signal when we've reached the target level
    let reached_target = Arc::new(AtomicBool::new(false));
    let reached_target_clone = reached_target.clone();

    let result = improver.improve(|result| {
        if result.level >= target_level {
            println!("\n✅ TARGET LEVEL {} REACHED!", target_level);
            println!("   Final counter: {}", result.counter);
            reached_target_clone.store(true, Ordering::SeqCst);
            // Stop searching when target level is reached
            false
        } else {
            // Continue searching
            true
        }
    });

    match result {
        Ok(_) => {
            if reached_target.load(Ordering::SeqCst) {
                println!("\n🎉 Successfully reached target security level {}!", target_level);
            } else {
                println!("\n⚠️  Stopped before reaching target level {}.", target_level);
            }
        }
        Err(e) => {
            eprintln!("\n❌ Error during improvement: {}", e);
            std::process::exit(1);
        }
    }

    // Display final statistics
    let stats = improver.get_statistics();
    print_statistics(stats.hashes_checked, stats.elapsed_secs);
}

fn run_improver_string<H: SecurityLevelHasher>(
    identity_str: &str,
    target_level: u8,
    hasher: H,
) {
    use std::time::{Duration, Instant};

    // Parse the identity
    let identity = match Ts3Identity::parse_identity(identity_str) {
        Ok(id) => id,
        Err(e) => {
            eprintln!("❌ Error parsing identity: {}", e);
            std::process::exit(1);
        }
    };

    let omega = identity.public_key_base64();
    let mut current_counter = identity.counter + 1;
    let mut best_level = identity.security_level();
    let mut best_counter = identity.counter;
    let mut hashes_checked: u64 = 0;
    let start_time = Instant::now();
    let mut last_print = Instant::now();

    const BATCH_SIZE: u64 = 10_000;

    println!("\nStarting level improvement search...");
    println!("Press Ctrl+C to stop\n");

    let reached_target = Arc::new(AtomicBool::new(false));

    loop {
        if reached_target.load(Ordering::SeqCst) {
            break;
        }

        // Check a batch of counters
        for _ in 0..BATCH_SIZE {
            let level = hasher.calculate_level(&omega, current_counter);
            hashes_checked += 1;

            if level > best_level {
                best_level = level;
                best_counter = current_counter;

                println!("\n🎉 NEW LEVEL FOUND!");
                println!("   Counter: {}", best_counter);
                println!("   Level: {}", best_level);
                println!("   Hashes checked: {}", format_number(hashes_checked));
                println!("   Updated identity string: {}V{}", best_counter, &identity_str[identity_str.find('V').unwrap() + 1..]);

                if best_level >= target_level {
                    println!("\n✅ TARGET LEVEL {} REACHED!", target_level);
                    println!("   Final counter: {}", best_counter);
                    reached_target.store(true, Ordering::SeqCst);
                    break;
                }
            }

            current_counter += 1;
        }

        // Print progress every second
        if last_print.elapsed() >= Duration::from_secs(1) {
            helpers::print_progress(best_level, current_counter, hashes_checked, start_time);
            last_print = Instant::now();
        }
    }

    if reached_target.load(Ordering::SeqCst) {
        println!("\n🎉 Successfully reached target security level {}!", target_level);
        println!("\n📋 Final identity string:");
        println!("   {}V{}", best_counter, &identity_str[identity_str.find('V').unwrap() + 1..]);
    } else {
        println!("\n⚠️  Stopped before reaching target level {}.", target_level);
    }

    // Display final statistics
    let elapsed_secs = start_time.elapsed().as_secs_f64();
    print_statistics(hashes_checked, elapsed_secs);
}
