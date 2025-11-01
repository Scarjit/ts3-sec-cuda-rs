# ts3-sec-cuda-rs

TeamSpeak 3 Security Level Tool - A Rust implementation for parsing and improving TS3 identity security levels using CPU-based SHA-1 hashing.

## Overview

This tool helps you work with TeamSpeak 3 identities by:
- **Decoding** identity files or strings to view counter, security level, and public key
- **Increasing** security levels through proof-of-work computation (finding better counter values)

Security levels are based on SHA-1 hash trailing zero bits, where level N proves ~2^N hashes were computed.

## Installation

```bash
cargo build --release
```

## Usage

### Decode Identity

Display information about an identity from a file:
```bash
cargo run -- decode --file inis/level8.ini
```

Or directly from an identity string:
```bash
cargo run -- decode --string '14VHjz+l0cOhfrqCaAhUCRm+9aXXlZENXNaWAMgByVlb3JQRB91flxmBCBDPxxmXUZ2EFdIf3twUVEWRW0ie2lzcHVFamdGP01VFgpJEQlTCRZYV31GE3J7AV8RBm9SAlphIWpMWk1zeFRzQUlnS2ExS1VuUnlMNnYxVWwvWFJaUVBieWc0a1FNRmlhcjlqTUVvM1pKUDNGcz0='
```

**Output:**
```
Identity Information:
  Counter:        14
  Security Level: 8
  Public Key:     ME0DAgcAAgEgAiEAy/hhqSBja7A6FTZG5s+BMnQfCqYyS9sGsbyMKBb7spYCIQCBEtZWrZtewnxuh2hsigJswGHchu3XcaiQDZziMsxTsA==

💡 Proof of Work:
  This identity proves ~256 SHA-1 hashes were computed.
```

### Increase Security Level

#### File Mode (saves progress to files)
Increases security level and creates new file `basename-<level>.ini`:
```bash
cargo run -- increase --file inis/level8.ini --target 12
```

**Behavior:**
- Creates new files: `level8-9.ini`, `level8-10.ini`, etc.
- Automatically deletes intermediate files (keeps only base and latest)
- Original file is never modified

#### String Mode (stdout only)
Increases security level from an identity string, logging only to stdout:
```bash
cargo run -- increase --string '14VHjz+l0cOhfrqCaAhUCRm+9aXXlZENXNaWAMgByVlb3JQRB91flxmBCBDPxxmXUZ2EFdIf3twUVEWRW0ie2lzcHVFamdGP01VFgpJEQlTCRZYV31GE3J7AV8RBm9SAlphIWpMWk1zeFRzQUlnS2ExS1VuUnlMNnYxVWwvWFJaUVBieWc0a1FNRmlhcjlqTUVvM1pKUDNGcz0=' --target 10
```

**Behavior:**
- No files created or modified
- Outputs updated identity string to stdout
- Perfect for testing or temporary operations

**Output example:**
```
🎉 NEW LEVEL FOUND!
   Counter: 201
   Level: 9
   Hashes checked: 187
   Updated identity string: 201V...

✅ TARGET LEVEL 9 REACHED!

📋 Final identity string:
   201VHjz+l0cOhfrqCaAhUCRm+9aXXlZENXNaWAMgByVlb3JQRB91flxmBCBDPxxmXUZ2EFdIf3twUVEWRW0ie2lzcHVFamdGP01VFgpJEQlTCRZYV31GE3J7AV8RBm9SAlphIWpMWk1zeFRzQUlnS2ExS1VuUnlMNnYxVWwvWFJaUVBieWc0a1FNRmlhcjlqTUVvM1pKUDNGcz0=

📊 Statistics:
   Total hashes checked: 187
   Average hashrate: 11.24 MH/s
   Time elapsed: 0.0s
```

### Options

```bash
cargo run -- --help                 # Show all commands
cargo run -- decode --help          # Show decode options
cargo run -- increase --help        # Show increase options
```

**Increase command options:**
- `--file <FILE>` - Path to identity.ini file
- `--string <STRING>` - Identity string directly
- `--target <LEVEL>` - Target security level (required)
- `--method <METHOD>` - Hasher method: `cpu` (default)

## Examples

**Quick test to level 10:**
```bash
cargo run -- increase --string '14VHjz+...' --target 10
```

**Production use with file:**
```bash
cargo run --release -- increase --file ~/.ts3client/identity.ini --target 20
```

**Verify your improved identity:**
```bash
cargo run -- decode --file ~/.ts3client/identity-20.ini
```

## Performance

CPU mode performance on test system (Linux 6.17.5-1-cachyos-eevdf):
- **~11-15 MH/s** (11-15 million hashes/second)
- Level 8→12: instant (< 1 second)
- Level 8→15: instant (< 1 second, ~8,000 hashes)
- Level 20+: minutes to hours depending on luck

The tool displays real-time hashrate and final statistics:
```
📊 Statistics:
   Total hashes checked: 8,045
   Average hashrate: 15.18 MH/s
   Time elapsed: 0.001s
```

Performance varies by CPU. Modern CPUs with good single-thread performance will achieve higher hashrates.

## Technical Details

- **Identity Format:** `COUNTER || 'V' || base64(obfuscate(base64(KEYPAIR_ASN1)))`
- **Security Level:** Number of trailing zero bits in `SHA-1(public_key || counter)`
- **Key Type:** P-256 ECC (secp256r1)
