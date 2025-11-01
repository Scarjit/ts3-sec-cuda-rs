// sha1_optimized.cu - Optimized SHA1 security level calculation
// OPTIMIZATIONS:
// - Fast path: Single-block SHA1 for short messages (total_len <= 55 bytes)
// - Slow path: Multi-block SHA1 using dynamic shared memory
// - Circular buffer for w[] (16 words instead of 80, reduces register pressure)
// - Shared memory for public key caching

#define ROTLEFT(a,b) (((a) << (b)) | ((a) >> (32-(b))))

// Device function: Convert u64 to ASCII string (decimal representation)
// Returns the number of digits written
__device__ int u64_to_string(unsigned long long value, unsigned char* buffer) {
    if (value == 0) {
        buffer[0] = '0';
        return 1;
    }

    // Temporary buffer to store digits in reverse order
    // 16 bytes is enough for counters up to 10^16 (10 quadrillion)
    // Max u64 needs 20 digits, but practical use cases don't exceed this
    unsigned char temp[16];
    int count = 0;

    // Extract digits one by one
    while (value > 0 && count < 16) {
        temp[count++] = '0' + (value % 10);
        value /= 10;
    }

    // Reverse digits into output buffer
    for (int i = 0; i < count; i++) {
        buffer[i] = temp[count - 1 - i];
    }

    return count;
}

// Device function: Count trailing zero bits in a 160-bit SHA1 hash
// Note: "trailing" here means from the beginning of the byte representation (byte 0 onwards)
__device__ unsigned char count_trailing_zero_bits(const unsigned int* hash) {
    unsigned char count = 0;

    // Hash is stored as 5 x 32-bit words
    // We need to convert to bytes and count from byte 0 (MSB) onwards
    for (int word_idx = 0; word_idx < 5; word_idx++) {
        unsigned int word = hash[word_idx];

        // Check each byte in the word (big-endian: MSB first)
        for (int byte_in_word = 0; byte_in_word < 4; byte_in_word++) {
            // Extract byte (MSB first for big-endian)
            unsigned char byte = (word >> (24 - byte_in_word * 8)) & 0xFF;

            if (byte == 0) {
                count += 8;
            } else {
                // Count trailing zeros in this byte
                unsigned char tz = 0;
                while (tz < 8 && (byte & (1u << tz)) == 0) {
                    tz++;
                }
                return count + tz;
            }
        }
    }

    return count;
}

// Device function: Perform SHA1 hash on a single block (64 bytes) with custom initial hash
// Optimized to reduce register pressure by computing w[] on-the-fly using circular buffer
__device__ void sha1_single_block_with_init(const unsigned int* block, unsigned int* hash_out, const unsigned int* init_hash) {
    // Use provided initial hash values (or SHA1 default if this is the first block)
    unsigned int h0 = init_hash[0];
    unsigned int h1 = init_hash[1];
    unsigned int h2 = init_hash[2];
    unsigned int h3 = init_hash[3];
    unsigned int h4 = init_hash[4];

    // Only store first 16 words to save registers (circular buffer)
    // w[] values for rounds 16-79 are computed on-the-fly and reuse the array
    unsigned int w[16];
    #pragma unroll
    for (int i = 0; i < 16; i++) {
        w[i] = block[i];
    }

    // SHA1 working variables
    unsigned int a = h0, b = h1, c = h2, d = h3, e = h4;

    // Rounds 0-19: f = (b & c) | ((~b) & d), k = 0x5A827999
    #pragma unroll
    for (int i = 0; i < 20; i++) {
        // Compute w[i] on-the-fly: for i < 16 use stored value, else compute expansion
        // Use modulo 16 indexing to reuse the w[] array as a circular buffer
        unsigned int wi = (i < 16) ? w[i] :
            ROTLEFT((w[(i-3)&15] ^ w[(i-8)&15] ^ w[(i-14)&15] ^ w[i&15]), 1);
        if (i >= 16) w[i&15] = wi;  // Store computed value back into circular buffer

        unsigned int f = (b & c) | ((~b) & d);
        unsigned int temp = ROTLEFT(a, 5) + f + e + 0x5A827999 + wi;
        e = d; d = c; c = ROTLEFT(b, 30); b = a; a = temp;
    }

    // Rounds 20-39: f = b ^ c ^ d, k = 0x6ED9EBA1
    #pragma unroll
    for (int i = 20; i < 40; i++) {
        // Inline w[] expansion with circular buffer indexing
        unsigned int wi = ROTLEFT((w[(i-3)&15] ^ w[(i-8)&15] ^ w[(i-14)&15] ^ w[i&15]), 1);
        w[i&15] = wi;

        unsigned int f = b ^ c ^ d;
        unsigned int temp = ROTLEFT(a, 5) + f + e + 0x6ED9EBA1 + wi;
        e = d; d = c; c = ROTLEFT(b, 30); b = a; a = temp;
    }

    // Rounds 40-59: f = (b & c) | (b & d) | (c & d), k = 0x8F1BBCDC
    #pragma unroll
    for (int i = 40; i < 60; i++) {
        // Inline w[] expansion with circular buffer indexing
        unsigned int wi = ROTLEFT((w[(i-3)&15] ^ w[(i-8)&15] ^ w[(i-14)&15] ^ w[i&15]), 1);
        w[i&15] = wi;

        unsigned int f = (b & c) | (b & d) | (c & d);
        unsigned int temp = ROTLEFT(a, 5) + f + e + 0x8F1BBCDC + wi;
        e = d; d = c; c = ROTLEFT(b, 30); b = a; a = temp;
    }

    // Rounds 60-79: f = b ^ c ^ d, k = 0xCA62C1D6
    #pragma unroll
    for (int i = 60; i < 80; i++) {
        // Inline w[] expansion with circular buffer indexing
        unsigned int wi = ROTLEFT((w[(i-3)&15] ^ w[(i-8)&15] ^ w[(i-14)&15] ^ w[i&15]), 1);
        w[i&15] = wi;

        unsigned int f = b ^ c ^ d;
        unsigned int temp = ROTLEFT(a, 5) + f + e + 0xCA62C1D6 + wi;
        e = d; d = c; c = ROTLEFT(b, 30); b = a; a = temp;
    }

    // Add to hash values
    hash_out[0] = h0 + a;
    hash_out[1] = h1 + b;
    hash_out[2] = h2 + c;
    hash_out[3] = h3 + d;
    hash_out[4] = h4 + e;
}

// Optimized kernel: Compute security levels for a range of counters
// Each thread handles one counter value
// DUAL-PATH OPTIMIZATION:
// - Fast path: Single-block SHA1 for short messages (total_len <= 55 bytes)
// - Slow path: Multi-block SHA1 using dynamic shared memory for longer messages
extern "C" __global__ void sha1_security_level_optimized(
    const unsigned char* public_key,   // Public key bytes in device memory
    int public_key_len,                // Length of public key
    unsigned long long start_counter,  // Starting counter value
    int num_counters,                  // Number of counters to process
    unsigned char* security_levels     // Output: security level for each counter
) {
    // Load public key into shared memory (fast L1 cache)
    // Shared memory is per-block and much faster than global device memory
    __shared__ unsigned char s_public_key[128];

    // Cooperative load: each thread in the block loads one byte at a time
    for (int i = threadIdx.x; i < public_key_len; i += blockDim.x) {
        s_public_key[i] = public_key[i];
    }
    // Synchronize to ensure all threads have completed loading before proceeding
    __syncthreads();

    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= num_counters) return;

    // Calculate this thread's counter value
    unsigned long long counter = start_counter + idx;

    // Convert counter to string FIRST (small 16-byte stack buffer)
    // This allows us to determine the total message length early
    // 16 bytes is enough for counters up to 10^16
    unsigned char counter_str[16];
    int counter_len = u64_to_string(counter, counter_str);

    int total_len = public_key_len + counter_len;

    // ============ FAST PATH: SINGLE BLOCK ============
    // If total message length <= 55 bytes, we can fit everything in one SHA1 block
    // Note: Most real-world public keys are longer and will use the slow path
    if (total_len <= 55) {
        // Build message directly into w[] array (single allocation for both input and expansion)
        // This eliminates the need for a separate block[] array (saves 64 bytes!)
        unsigned int w[16];
        for (int i = 0; i < 16; i++) w[i] = 0;

        int pos = 0;

        // Pack public key directly into w[] words (big-endian format)
        for (int i = 0; i < public_key_len; i++) {
            int word_idx = pos / 4;      // Which 32-bit word
            int byte_idx = pos % 4;      // Which byte in that word (0-3)
            // Shift byte to correct position (big-endian: MSB first)
            w[word_idx] |= ((unsigned int)s_public_key[i]) << (24 - byte_idx * 8);
            pos++;
        }

        // Pack counter string directly into w[] words
        for (int i = 0; i < counter_len; i++) {
            int word_idx = pos / 4;
            int byte_idx = pos % 4;
            w[word_idx] |= ((unsigned int)counter_str[i]) << (24 - byte_idx * 8);
            pos++;
        }

        // Add SHA1 padding: 0x80 byte followed by zeros, then length
        int word_idx = pos / 4;
        int byte_idx = pos % 4;
        w[word_idx] |= 0x80u << (24 - byte_idx * 8);  // Append 0x80 byte

        // Add message length in bits at the end (words 14-15)
        unsigned long long bit_len = (unsigned long long)total_len * 8;
        w[14] = (unsigned int)(bit_len >> 32);  // High 32 bits
        w[15] = (unsigned int)(bit_len & 0xFFFFFFFF);  // Low 32 bits

        // Compute SHA1 hash (single block from scratch - no init_hash array needed)
        // SHA1 initial hash values (constants)
        unsigned int h0 = 0x67452301;
        unsigned int h1 = 0xEFCDAB89;
        unsigned int h2 = 0x98BADCFE;
        unsigned int h3 = 0x10325476;
        unsigned int h4 = 0xC3D2E1F0;

        // SHA1 working variables
        unsigned int a = h0, b = h1, c = h2, d = h3, e = h4;

        // Rounds 0-19
        #pragma unroll
        for (int i = 0; i < 20; i++) {
            unsigned int wi = (i < 16) ? w[i] :
                ROTLEFT((w[(i-3)&15] ^ w[(i-8)&15] ^ w[(i-14)&15] ^ w[i&15]), 1);
            if (i >= 16) w[i&15] = wi;

            unsigned int f = (b & c) | ((~b) & d);
            unsigned int temp = ROTLEFT(a, 5) + f + e + 0x5A827999 + wi;
            e = d; d = c; c = ROTLEFT(b, 30); b = a; a = temp;
        }

        // Rounds 20-39
        #pragma unroll
        for (int i = 20; i < 40; i++) {
            unsigned int wi = ROTLEFT((w[(i-3)&15] ^ w[(i-8)&15] ^ w[(i-14)&15] ^ w[i&15]), 1);
            w[i&15] = wi;

            unsigned int f = b ^ c ^ d;
            unsigned int temp = ROTLEFT(a, 5) + f + e + 0x6ED9EBA1 + wi;
            e = d; d = c; c = ROTLEFT(b, 30); b = a; a = temp;
        }

        // Rounds 40-59
        #pragma unroll
        for (int i = 40; i < 60; i++) {
            unsigned int wi = ROTLEFT((w[(i-3)&15] ^ w[(i-8)&15] ^ w[(i-14)&15] ^ w[i&15]), 1);
            w[i&15] = wi;

            unsigned int f = (b & c) | (b & d) | (c & d);
            unsigned int temp = ROTLEFT(a, 5) + f + e + 0x8F1BBCDC + wi;
            e = d; d = c; c = ROTLEFT(b, 30); b = a; a = temp;
        }

        // Rounds 60-79
        #pragma unroll
        for (int i = 60; i < 80; i++) {
            unsigned int wi = ROTLEFT((w[(i-3)&15] ^ w[(i-8)&15] ^ w[(i-14)&15] ^ w[i&15]), 1);
            w[i&15] = wi;

            unsigned int f = b ^ c ^ d;
            unsigned int temp = ROTLEFT(a, 5) + f + e + 0xCA62C1D6 + wi;
            e = d; d = c; c = ROTLEFT(b, 30); b = a; a = temp;
        }

        // Compute final hash values
        h0 += a;
        h1 += b;
        h2 += c;
        h3 += d;
        h4 += e;

        // Count trailing zero bits inline (no hash array needed!)
        unsigned char count = 0;
        for (int word_idx = 0; word_idx < 5; word_idx++) {
            unsigned int word = (word_idx == 0) ? h0 :
                               (word_idx == 1) ? h1 :
                               (word_idx == 2) ? h2 :
                               (word_idx == 3) ? h3 : h4;

            for (int byte_in_word = 0; byte_in_word < 4; byte_in_word++) {
                unsigned char byte = (word >> (24 - byte_in_word * 8)) & 0xFF;

                if (byte == 0) {
                    count += 8;
                } else {
                    unsigned char tz = 0;
                    while (tz < 8 && (byte & (1u << tz)) == 0) {
                        tz++;
                    }
                    security_levels[idx] = count + tz;
                    return;
                }
            }
        }

        security_levels[idx] = count;
        return;  // Fast path complete!
    }

    // ============ SLOW PATH: MULTI-BLOCK ============
    // For messages > 55 bytes, use dynamic shared memory to avoid stack usage
    // Dynamic shared memory is allocated per-block at kernel launch
    extern __shared__ unsigned char s_message[];
    unsigned char* my_message = &s_message[threadIdx.x * 128];

    // Build message in dynamic shared memory
    int msg_len = 0;
    for (int i = 0; i < public_key_len; i++) {
        my_message[msg_len++] = s_public_key[i];
    }
    for (int i = 0; i < counter_len; i++) {
        my_message[msg_len++] = counter_str[i];
    }

    // Multi-block SHA1 processing
    unsigned int hash[5] = {0x67452301, 0xEFCDAB89, 0x98BADCFE, 0x10325476, 0xC3D2E1F0};
    unsigned long long bit_len = (unsigned long long)msg_len * 8;

    // Calculate number of blocks needed
    int blocks_needed = (msg_len + 1 + 8 + 63) / 64;

    // Process complete 64-byte blocks
    int pos = 0;
    while (pos + 64 <= msg_len) {
        unsigned int block[16];

        // Convert 64 bytes to 16 words (big-endian)
        for (int i = 0; i < 16; i++) {
            block[i] = ((unsigned int)my_message[pos + i*4 + 0] << 24) |
                       ((unsigned int)my_message[pos + i*4 + 1] << 16) |
                       ((unsigned int)my_message[pos + i*4 + 2] << 8) |
                       ((unsigned int)my_message[pos + i*4 + 3]);
        }

        sha1_single_block_with_init(block, hash, hash);
        pos += 64;
    }

    // Handle final block(s) with padding
    int remaining = msg_len - pos;
    int final_blocks = blocks_needed - (pos / 64);

    for (int block_num = 0; block_num < final_blocks; block_num++) {
        unsigned int block[16];
        for (int i = 0; i < 16; i++) block[i] = 0;

        // Copy remaining message bytes for first final block
        if (block_num == 0) {
            for (int i = 0; i < remaining; i++) {
                int word_idx = i / 4;
                int byte_idx = i % 4;
                block[word_idx] |= ((unsigned int)my_message[pos + i]) << (24 - byte_idx * 8);
            }

            // Add 0x80 byte
            int padding_byte = remaining;
            int word_idx = padding_byte / 4;
            int byte_idx = padding_byte % 4;
            block[word_idx] |= 0x80u << (24 - byte_idx * 8);
        }

        // Add length to final block
        if (block_num == final_blocks - 1) {
            block[14] = (unsigned int)(bit_len >> 32);
            block[15] = (unsigned int)(bit_len & 0xFFFFFFFF);
        }

        sha1_single_block_with_init(block, hash, hash);
    }

    // Count trailing zero bits and write result
    security_levels[idx] = count_trailing_zero_bits(hash);
}
