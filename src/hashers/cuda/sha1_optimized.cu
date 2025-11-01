// sha1_optimized.cu
// Optimized CUDA kernel for TS3 security level calculation
// - Generates counter strings on GPU
// - No CPU-side string allocations
// - Directly outputs security levels (trailing zero bits)
// - Optimized with intrinsics and loop unrolling

#define ROTLEFT(a,b) (((a) << (b)) | ((a) >> (32-(b))))

// Device function: Convert u64 to ASCII string (decimal representation)
// Returns the number of digits written
// Optimized version using fewer divisions by processing digits in chunks of 4
__device__ int u64_to_string(unsigned long long value, unsigned char* buffer) {
    if (value == 0) {
        buffer[0] = '0';
        return 1;
    }

    // Convert using fast division by powers of 10
    // Temporary buffer to store digits in reverse order
    unsigned char temp[20];  // Max digits for u64 is 20
    int count = 0;

    // Unrolled division: process 4 digits at a time to reduce expensive division ops
    // Division by 10000 is much faster than four separate divisions by 10
    while (value >= 10000) {
        unsigned long long q = value / 10000;  // Quotient
        unsigned int r = value - q * 10000;     // Remainder (last 4 digits)

        // Extract 4 digits from remainder (stored in reverse)
        temp[count++] = '0' + (r % 10); r /= 10;
        temp[count++] = '0' + (r % 10); r /= 10;
        temp[count++] = '0' + (r % 10); r /= 10;
        temp[count++] = '0' + r;

        value = q;  // Continue with quotient
    }

    // Handle remaining digits (< 10000) using standard division
    while (value > 0) {
        temp[count++] = '0' + (value % 10);
        value /= 10;
    }

    // Reverse digits into output buffer (stored in reverse in temp)
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
    // Word 0 = bytes 0-3, Word 1 = bytes 4-7, etc.
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
                // trailing_zeros counts from LSB, which is what we want
                unsigned char tz = 0;
                while (tz < 8 && (byte & (1u << tz)) == 0) {
                    tz++;
                }
                count += tz;
                return count;
            }
        }
    }

    return count;
}

// Device function: Perform SHA1 hash on a single block (64 bytes) with custom initial hash
// Optimized to reduce register pressure by computing w[] on-the-fly
__device__ void sha1_single_block_with_init(const unsigned int* block, unsigned int* hash_out, const unsigned int* init_hash) {
    // Use provided initial hash values (or SHA1 default if this is the first block)
    unsigned int h0 = init_hash[0];
    unsigned int h1 = init_hash[1];
    unsigned int h2 = init_hash[2];
    unsigned int h3 = init_hash[3];
    unsigned int h4 = init_hash[4];

    // Only store first 16 words to save registers (was 80, now 16)
    // w[] values for rounds 16-79 are computed on-the-fly and reuse the array
    // This trades extra computation for reduced register pressure on GPU
    unsigned int w[16];
    for (int i = 0; i < 16; i++) {
        w[i] = block[i];
    }

    // Main loop (80 rounds) - unrolled into 4 loops to eliminate branches
    unsigned int a = h0, b = h1, c = h2, d = h3, e = h4;
    unsigned int f, temp;

    // Rounds 0-19: f = (b & c) | ((~b) & d), k = 0x5A827999
    #pragma unroll
    for (int i = 0; i < 20; i++) {
        // Compute w[i] on-the-fly: for i < 16 use stored value, else compute expansion
        // Use modulo 16 indexing to reuse the w[] array as a circular buffer
        unsigned int wi = (i < 16) ? w[i] :
            ROTLEFT((w[(i-3)&15] ^ w[(i-8)&15] ^ w[(i-14)&15] ^ w[i&15]), 1);
        if (i >= 16) w[i&15] = wi;  // Store computed value back into circular buffer

        f = (b & c) | ((~b) & d);
        temp = ROTLEFT(a, 5) + f + e + 0x5A827999 + wi;
        e = d;
        d = c;
        c = ROTLEFT(b, 30);
        b = a;
        a = temp;
    }

    // Rounds 20-39: f = b ^ c ^ d, k = 0x6ED9EBA1
    #pragma unroll
    for (int i = 20; i < 40; i++) {
        // Inline w[] expansion with circular buffer indexing
        unsigned int wi = ROTLEFT((w[(i-3)&15] ^ w[(i-8)&15] ^ w[(i-14)&15] ^ w[i&15]), 1);
        w[i&15] = wi;

        f = b ^ c ^ d;
        temp = ROTLEFT(a, 5) + f + e + 0x6ED9EBA1 + wi;
        e = d;
        d = c;
        c = ROTLEFT(b, 30);
        b = a;
        a = temp;
    }

    // Rounds 40-59: f = (b & c) | (b & d) | (c & d), k = 0x8F1BBCDC
    #pragma unroll
    for (int i = 40; i < 60; i++) {
        // Inline w[] expansion with circular buffer indexing
        unsigned int wi = ROTLEFT((w[(i-3)&15] ^ w[(i-8)&15] ^ w[(i-14)&15] ^ w[i&15]), 1);
        w[i&15] = wi;

        f = (b & c) | (b & d) | (c & d);
        temp = ROTLEFT(a, 5) + f + e + 0x8F1BBCDC + wi;
        e = d;
        d = c;
        c = ROTLEFT(b, 30);
        b = a;
        a = temp;
    }

    // Rounds 60-79: f = b ^ c ^ d, k = 0xCA62C1D6
    #pragma unroll
    for (int i = 60; i < 80; i++) {
        // Inline w[] expansion with circular buffer indexing
        unsigned int wi = ROTLEFT((w[(i-3)&15] ^ w[(i-8)&15] ^ w[(i-14)&15] ^ w[i&15]), 1);
        w[i&15] = wi;

        f = b ^ c ^ d;
        temp = ROTLEFT(a, 5) + f + e + 0xCA62C1D6 + wi;
        e = d;
        d = c;
        c = ROTLEFT(b, 30);
        b = a;
        a = temp;
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
    // This efficiently utilizes memory bandwidth and ensures coalesced access
    for (int i = threadIdx.x; i < public_key_len; i += blockDim.x) {
        s_public_key[i] = public_key[i];
    }
    // Synchronize to ensure all threads have completed loading before proceeding
    __syncthreads();

    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= num_counters) return;

    // Calculate this thread's counter value
    unsigned long long counter = start_counter + idx;

    // Build message: public_key + counter_string
    unsigned char message[128];  // Reduced stack: 108 (key) + 20 (max counter) = 128
    int msg_len = 0;

    // Copy public key from shared memory (much faster than device memory!)
    // Each thread reads from the cached copy instead of global memory
    for (int i = 0; i < public_key_len; i++) {
        message[msg_len++] = s_public_key[i];
    }

    // Convert counter to string and append
    int counter_len = u64_to_string(counter, message + msg_len);
    msg_len += counter_len;

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
            block[i] = ((unsigned int)message[pos + i*4 + 0] << 24) |
                       ((unsigned int)message[pos + i*4 + 1] << 16) |
                       ((unsigned int)message[pos + i*4 + 2] << 8) |
                       ((unsigned int)message[pos + i*4 + 3]);
        }

        // Process this block
        sha1_single_block_with_init(block, hash, hash);
        pos += 64;
    }

    // Handle final block(s) with padding
    int remaining = msg_len - pos;
    int final_blocks = blocks_needed - (pos / 64);

    for (int block_num = 0; block_num < final_blocks; block_num++) {
        unsigned int block[16];

        // Initialize to zero
        for (int i = 0; i < 16; i++) {
            block[i] = 0;
        }

        // Copy remaining message bytes for first final block
        if (block_num == 0) {
            for (int i = 0; i < remaining; i++) {
                int word_idx = i / 4;
                int byte_idx = i % 4;
                block[word_idx] |= ((unsigned int)message[pos + i]) << (24 - byte_idx * 8);
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

    // Count trailing zero bits
    security_levels[idx] = count_trailing_zero_bits(hash);
}
