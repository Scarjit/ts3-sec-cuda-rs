// sha1.cu
#include <stdint.h>

#define ROTLEFT(a,b) (((a) << (b)) | ((a) >> (32-(b))))

extern "C" __global__ void sha1_simple(
    const uint32_t* inputs,  // Input data (16 words per hash)
    uint32_t* outputs,        // Output hashes (5 words per hash)
    int num_hashes
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= num_hashes) return;

    // SHA1 initial hash values
    uint32_t h0 = 0x67452301;
    uint32_t h1 = 0xEFCDAB89;
    uint32_t h2 = 0x98BADCFE;
    uint32_t h3 = 0x10325476;
    uint32_t h4 = 0xC3D2E1F0;

    // Get input block for this thread (16 words = 64 bytes)
    const uint32_t* block = &inputs[idx * 16];

    // Expand to 80 words
    uint32_t w[80];
    for (int i = 0; i < 16; i++) {
        w[i] = block[i];
    }
    for (int i = 16; i < 80; i++) {
        w[i] = ROTLEFT((w[i-3] ^ w[i-8] ^ w[i-14] ^ w[i-16]), 1);
    }

    // Main loop (80 rounds)
    uint32_t a = h0, b = h1, c = h2, d = h3, e = h4;
    uint32_t f, k, temp;

    for (int i = 0; i < 80; i++) {
        if (i < 20) {
            f = (b & c) | ((~b) & d);
            k = 0x5A827999;
        } else if (i < 40) {
            f = b ^ c ^ d;
            k = 0x6ED9EBA1;
        } else if (i < 60) {
            f = (b & c) | (b & d) | (c & d);
            k = 0x8F1BBCDC;
        } else {
            f = b ^ c ^ d;
            k = 0xCA62C1D6;
        }

        temp = ROTLEFT(a, 5) + f + e + k + w[i];
        e = d;
        d = c;
        c = ROTLEFT(b, 30);
        b = a;
        a = temp;
    }

    // Add to hash values
    h0 += a;
    h1 += b;
    h2 += c;
    h3 += d;
    h4 += e;

    // Store result (160-bit hash = 5 x 32-bit words)
    outputs[idx * 5 + 0] = h0;
    outputs[idx * 5 + 1] = h1;
    outputs[idx * 5 + 2] = h2;
    outputs[idx * 5 + 3] = h3;
    outputs[idx * 5 + 4] = h4;
}