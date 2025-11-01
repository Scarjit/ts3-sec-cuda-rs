// sha1.cu
// Using native CUDA types (unsigned int = 32-bit)
#define ROTLEFT(a,b) (((a) << (b)) | ((a) >> (32-(b))))

extern "C" __global__ void sha1_simple(
    const unsigned int* inputs,    // Input data (16 words per hash)
    unsigned int* outputs,          // Output hashes (5 words per hash)
    int num_hashes,
    const unsigned int* init_hash  // Initial hash values (5 words)
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= num_hashes) return;

    // SHA1 initial hash values from parameter
    unsigned int h0 = init_hash[0];
    unsigned int h1 = init_hash[1];
    unsigned int h2 = init_hash[2];
    unsigned int h3 = init_hash[3];
    unsigned int h4 = init_hash[4];

    // Get input block for this thread (16 words = 64 bytes)
    const unsigned int* block = &inputs[idx * 16];

    // Expand to 80 words
    unsigned int w[80];
    for (int i = 0; i < 16; i++) {
        w[i] = block[i];
    }
    for (int i = 16; i < 80; i++) {
        w[i] = ROTLEFT((w[i-3] ^ w[i-8] ^ w[i-14] ^ w[i-16]), 1);
    }

    // Main loop (80 rounds)
    unsigned int a = h0, b = h1, c = h2, d = h3, e = h4;
    unsigned int f, k, temp;

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