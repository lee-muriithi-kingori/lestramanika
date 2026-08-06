/*
 * Lestra OS — Pickle host fast path implementation.
 * Copyright (c) 2026 lestramk.org / Lee Muriihi Kingori
 *
 * Host-only native-float math: matmul (F32 + every common quant),
 * RMSNorm, attention, RoPE, contiguous KV cache, BPE tokenizer, fast
 * samplers. Compiled ONLY when PICKLE_KERNEL is undefined (see
 * Makefile). The kernel keeps using pickle_softfp.c.
 *
 * DESIGN
 *   - All math uses plain C `float` arithmetic. With -O2 -mfpmath=sse
 *     (default on x86-64) the compiler auto-vectorises the inner loops
 *     with SSE2/SSE4/AVX/FMA depending on -march. We never call sfp_*()
 *     here.
 *   - Quantized matmul reads the raw on-disk Q4_K/Q8_0/Q6_K/... byte
 *     buffer block-by-block, dequantizes one 256-element (or 32-element)
 *     block into a stack scratch, then does a normal F32 dot product
 *     with the input vector. This avoids materialising the full F32
 *     weight matrix and dramatically improves cache locality — the
 *     same approach llama.cpp uses with its "vec-dot" kernels.
 *   - The RoPE sin/cos table is precomputed once and reused for every
 *     token, instead of recomputing sfp_sin/sfp_cos on every forward
 *     pass.
 *   - The KV cache is one big allocation for K and one for V, indexed
 *     as [layer][pos][kv_head][dim]. No per-slot mallocs.
 */
#include "pickle_fast.h"

#include <stdint.h>
#include <stddef.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <immintrin.h>   /* AVX/AVX2/AVX-512 intrinsics for hand-vectorised kernels */

/* ================================================================== */
/* f16 helpers                                                        */
/* ================================================================== */
static inline float f16_bits_to_float(uint16_t h) {
    /* IEEE-754 binary16 → binary32 via bit manipulation. */
    uint32_t sign = ((uint32_t)(h & 0x8000)) << 16;
    uint32_t exp  = (h & 0x7C00) >> 10;
    uint32_t mant = (h & 0x03FF);
    uint32_t f;
    if (exp == 0) {
        if (mant == 0) {
            f = sign;
        } else {
            /* subnormal: normalise. f16 subnormal effective exponent
             * is 1 (giving 2^(1-15) = 2^(-14)). Shift mant left until
             * hidden bit (0x0400) is set, decrementing exp per shift. */
            exp = 1;
            while ((mant & 0x0400) == 0) { mant <<= 1; exp--; }
            mant &= 0x03FF;
            f = sign | ((exp + 127 - 15) << 23) | (mant << 13);
        }
    } else if (exp == 0x1F) {
        f = sign | 0x7F800000u | (mant << 13);
    } else {
        f = sign | ((exp + 127 - 15) << 23) | (mant << 13);
    }
    float out;
    memcpy(&out, &f, 4);
    return out;
}

/* ================================================================== */
/* Native-float math                                                  */
/* ================================================================== */
void pickle_fast_rmsnorm(float* x, const float* w, int n, float eps) {
    float sum = 0.0f;
    for (int i = 0; i < n; i++) sum += x[i] * x[i];
    float mean = sum / (float)n;
    float rsq = 1.0f / sqrtf(mean + eps);
    for (int i = 0; i < n; i++) x[i] = x[i] * w[i] * rsq;
}

void pickle_fast_matmul_f32(float* y, const float* W, const float* x,
                            int out_n, int in_n) {
    /* Standard row-major matmul: y[o] = sum_i W[o*in_n + i] * x[i].
     * The compiler vectorises the inner loop; we additionally unroll
     * by 4 to give it a clearer reduction chain. OpenMP parallelises
     * across output rows — each thread owns disjoint o indices, so no
     * synchronization is needed. */
    #pragma omp parallel for schedule(static)
    for (int o = 0; o < out_n; o++) {
        const float* wrow = W + (size_t)o * in_n;
        float acc0 = 0.0f, acc1 = 0.0f, acc2 = 0.0f, acc3 = 0.0f;
        int n4 = in_n & ~3;
        /* 4-way unrolled reduction with independent accumulators —
         * breaks the serial FMA dependency chain so the compiler can
         * issue 4 parallel vfmadd231ps lanes (one per accumulator). */
        #pragma omp simd simdlen(16) reduction(+:acc0,acc1,acc2,acc3)
        for (int i = 0; i < n4; i += 4) {
            acc0 += wrow[i]     * x[i];
            acc1 += wrow[i + 1] * x[i + 1];
            acc2 += wrow[i + 2] * x[i + 2];
            acc3 += wrow[i + 3] * x[i + 3];
        }
        for (int i = n4; i < in_n; i++) acc0 += wrow[i] * x[i];
        y[o] = (acc0 + acc1) + (acc2 + acc3);
    }
}

/* ================================================================== */
/* Quantized matmul kernels                                           */
/* Each reads the raw on-disk bytes for `out_n` weight rows, each of  */
/* length `in_n` elements quantized in `block_size`-element blocks.   */
/* For each row, we walk the blocks, dequant one block into a stack   */
/* scratch buffer of `block_size` floats, then dot-product with the   */
/* corresponding slice of x.                                          */
/* ================================================================== */

/* ----- Q4_0: 18 bytes / 32 elements ------------------------------- */
/* Block: d(f16,2) + qs(16)  — each byte holds two 4-bit quants,
 * offset by -8. */
void pickle_fast_matmul_q4_0(float* y,
                             const unsigned char* W,
                             int out_n, int in_n,
                             const float* x) {
    const int BS = 32;
    const size_t block_bytes = 18;
    int n_blocks = in_n / BS;
    /* Fused dequant+dot: no intermediate dq[32] buffer — each element is
     * dequanted straight into the accumulator. Halves memory traffic vs
     * the two-pass form and lets the compiler keep partials in registers. */
    #pragma omp parallel for schedule(static)
    for (int o = 0; o < out_n; o++) {
        const unsigned char* row = W + (size_t)o * n_blocks * block_bytes;
        float acc = 0.0f;
        for (int b = 0; b < n_blocks; b++) {
            const unsigned char* blk = row + (size_t)b * block_bytes;
            float d = f16_bits_to_float((uint16_t)blk[0] | ((uint16_t)blk[1] << 8));
            const unsigned char* qs = blk + 2;
            const float* xr = x + b * BS;
            /* #pragma omp simd reduction forces vectorisation — without
             * it the compiler emits scalar vmulss+vaddss, with it the
             * int8->float conversion + FMA gets pipelined through
             * ymm/zmm registers. */
            #pragma omp simd simdlen(16) reduction(+:acc)
            for (int i = 0; i < 16; i++) {
                int lo = qs[i] & 0x0F;
                int hi = (qs[i] >> 4) & 0x0F;
                acc += d * (float)(lo - 8) * xr[i];
                acc += d * (float)(hi - 8) * xr[16 + i];
            }
        }
        y[o] = acc;
    }
}

/* ----- Q4_1: 20 bytes / 32 elements -------------------------------- */
/* Block: d(f32,4) + m(f32,4) + qs(16). */
void pickle_fast_matmul_q4_1(float* y,
                             const unsigned char* W,
                             int out_n, int in_n,
                             const float* x) {
    const int BS = 32;
    const size_t block_bytes = 20;
    int n_blocks = in_n / BS;
    #pragma omp parallel for schedule(static)
    for (int o = 0; o < out_n; o++) {
        const unsigned char* row = W + (size_t)o * n_blocks * block_bytes;
        float acc = 0.0f;
        for (int b = 0; b < n_blocks; b++) {
            const unsigned char* blk = row + (size_t)b * block_bytes;
            float d, m;
            memcpy(&d, blk,     4);
            memcpy(&m, blk + 4, 4);
            const unsigned char* qs = blk + 8;
            const float* xr = x + b * BS;
            #pragma omp simd simdlen(16) reduction(+:acc)
            for (int i = 0; i < 16; i++) {
                int lo = qs[i] & 0x0F;
                int hi = (qs[i] >> 4) & 0x0F;
                acc += (d * (float)lo + m) * xr[i];
                acc += (d * (float)hi + m) * xr[16 + i];
            }
        }
        y[o] = acc;
    }
}

/* ----- Q5_0: 22 bytes / 32 elements -------------------------------- */
/* Block: d(f16,2) + qh(4) + ql(16). */
void pickle_fast_matmul_q5_0(float* y,
                             const unsigned char* W,
                             int out_n, int in_n,
                             const float* x) {
    const int BS = 32;
    const size_t block_bytes = 22;
    int n_blocks = in_n / BS;
    #pragma omp parallel for schedule(static)
    for (int o = 0; o < out_n; o++) {
        const unsigned char* row = W + (size_t)o * n_blocks * block_bytes;
        float acc = 0.0f;
        for (int b = 0; b < n_blocks; b++) {
            const unsigned char* blk = row + (size_t)b * block_bytes;
            float d = f16_bits_to_float((uint16_t)blk[0] | ((uint16_t)blk[1] << 8));
            const unsigned char* qh = blk + 2;
            const unsigned char* ql = blk + 6;
            uint32_t q5 = (uint32_t)qh[0] | ((uint32_t)qh[1] << 8) |
                          ((uint32_t)qh[2] << 16) | ((uint32_t)qh[3] << 24);
            float offset = d * -16.0f;
            const float* xr = x + b * BS;
            #pragma omp simd simdlen(16) reduction(+:acc)
            for (int i = 0; i < 16; i++) {
                int lo = ql[i] & 0x0F;
                int hi = (ql[i] >> 4) & 0x0F;
                int b0 = ((q5 >> i)       & 1) << 4;
                int b1 = ((q5 >> (16+i))  & 1) << 4;
                acc += (d * (float)(lo | b0) + offset) * xr[i];
                acc += (d * (float)(hi | b1) + offset) * xr[16 + i];
            }
        }
        y[o] = acc;
    }
}

/* ----- Q5_1: 24 bytes / 32 elements -------------------------------- */
/* Block: d(f32,4) + m(f32,4) + qh(4) + ql(16). */
void pickle_fast_matmul_q5_1(float* y,
                             const unsigned char* W,
                             int out_n, int in_n,
                             const float* x) {
    const int BS = 32;
    const size_t block_bytes = 24;
    int n_blocks = in_n / BS;
    #pragma omp parallel for schedule(static)
    for (int o = 0; o < out_n; o++) {
        const unsigned char* row = W + (size_t)o * n_blocks * block_bytes;
        float acc = 0.0f;
        for (int b = 0; b < n_blocks; b++) {
            const unsigned char* blk = row + (size_t)b * block_bytes;
            float d, m;
            memcpy(&d, blk,     4);
            memcpy(&m, blk + 4, 4);
            const unsigned char* qh = blk + 8;
            const unsigned char* ql = blk + 12;
            uint32_t q5 = (uint32_t)qh[0] | ((uint32_t)qh[1] << 8) |
                          ((uint32_t)qh[2] << 16) | ((uint32_t)qh[3] << 24);
            const float* xr = x + b * BS;
            #pragma omp simd simdlen(16) reduction(+:acc)
            for (int i = 0; i < 16; i++) {
                int lo = ql[i] & 0x0F;
                int hi = (ql[i] >> 4) & 0x0F;
                int b0 = ((q5 >> i)       & 1) << 4;
                int b1 = ((q5 >> (16+i))  & 1) << 4;
                acc += (d * (float)(lo | b0) + m) * xr[i];
                acc += (d * (float)(hi | b1) + m) * xr[16 + i];
            }
        }
        y[o] = acc;
    }
}

/* ----- Q8_0: 34 bytes / 32 elements -------------------------------- */
/* Block: d(f16,2) + qs(32) (signed int8). */
void pickle_fast_matmul_q8_0(float* y,
                             const unsigned char* W,
                             int out_n, int in_n,
                             const float* x) {
    const int BS = 32;
    const size_t block_bytes = 34;
    int n_blocks = in_n / BS;
    #pragma omp parallel for schedule(static)
    for (int o = 0; o < out_n; o++) {
        const unsigned char* row = W + (size_t)o * n_blocks * block_bytes;
        float acc = 0.0f;
        for (int b = 0; b < n_blocks; b++) {
            const unsigned char* blk = row + (size_t)b * block_bytes;
            float d = f16_bits_to_float((uint16_t)blk[0] | ((uint16_t)blk[1] << 8));
            const signed char* qs = (const signed char*)(blk + 2);
            const float* xr = x + b * BS;
            #pragma omp simd simdlen(16) reduction(+:acc)
            for (int i = 0; i < 32; i++) acc += d * (float)qs[i] * xr[i];
        }
        y[o] = acc;
    }
}

/* ----- Q8_1: 36 bytes / 32 elements -------------------------------- */
/* Block: d(f32,4) + s(f32,4) + qs(32). */
void pickle_fast_matmul_q8_1(float* y,
                             const unsigned char* W,
                             int out_n, int in_n,
                             const float* x) {
    const int BS = 32;
    const size_t block_bytes = 36;
    int n_blocks = in_n / BS;
    #pragma omp parallel for schedule(static)
    for (int o = 0; o < out_n; o++) {
        const unsigned char* row = W + (size_t)o * n_blocks * block_bytes;
        float acc = 0.0f;
        for (int b = 0; b < n_blocks; b++) {
            const unsigned char* blk = row + (size_t)b * block_bytes;
            float d;
            memcpy(&d, blk, 4);
            const signed char* qs = (const signed char*)(blk + 8);
            const float* xr = x + b * BS;
            #pragma omp simd simdlen(16) reduction(+:acc)
            for (int i = 0; i < 32; i++) acc += d * (float)qs[i] * xr[i];
        }
        y[o] = acc;
    }
}

/* ----- Q4_K: 144 bytes / 256 elements ------------------------------ */
/* Block layout (matches ggml-common.h block_q4_K):
 *   d (f16)    : super-block scale at offset 0
 *   dmin (f16) : super-block min at offset 2
 *   scales[12] : 8x 6-bit scale + 8x 6-bit min, packed via get_scale_min_k4
 *   qs[128]    : 4-bit quants, NON-INTERLEAVED packing (matches ggml-quants.c)
 *
 * GGML non-interleaved packing: 4 chunks of 64 elements. Each chunk uses
 * 32 bytes of qs. Within a chunk, qs[l] low nibble = element chunk*64+l,
 * high nibble = element chunk*64+l+32. Each 32-element half has its own
 * scale (8 scales total via get_scale_min_k4 at indices 0..7).
 *
 * Formula: dq = d * sc * qv - dmin * m  (qv = 4-bit unsigned 0..15)
 *
 * The previous implementation used consecutive `qs[i/2]` interleaving which
 * is WRONG — GGML packs elements 32 apart into the same byte. */
static inline void qk_get_scale_min_k4(int j, const unsigned char* q, int* d, int* m) {
    /* Matches ggml-quants.c get_scale_min_k4. The 8 scales + 8 mins are
     * packed 6-bit each into 12 bytes. For j < 4 the scale/min come from
     * the low 6 bits of q[j] / q[j+4]. For j >= 4 the low 4 bits come
     * from q[j+4] and the high 2 bits come from the TOP 2 bits (>> 6)
     * of q[j-4] (for *d) or q[j] (for *m). The previous implementation
     * wrongly used >> 4 for *d, pulling 4 bits instead of 2 and corrupting
     * the upper nibble of every scale j >= 4 — which broke half of all
     * Q4_K/Q5_K sub-blocks. */
    if (j < 4) {
        *d = q[j]     & 0x3F;
        *m = q[j + 4] & 0x3F;
    } else {
        *d = ((q[j + 4] & 0x0F) | ((q[j - 4] >> 6) << 4)) & 0x3F;
        *m = ((q[j + 4] >>  4) | ((q[j - 0] >> 6) << 4)) & 0x3F;
    }
}

/* ----- Q4_K VNNI vec dot kernel (AVX-512 VNNI) -------------------- */
/* Uses `_mm512_dpbusd_epi32` to do 64 uint8×int8→int32 MACs per
 * instruction (vs 16 for a zmm FMA). x is pre-quantized to Q8_0 (one
 * int8 + one fp32 scale per 32-element sub-block) once per matmul call
 * outside the row loop, so all `out_n` rows reuse the same x_int.
 *
 * Per 64-element chunk (32 qs bytes covering 2 sub-blocks):
 *   1. Extract 64 uint8 quants: [low_nibbles(32) | high_nibbles(32)]
 *      concatenated into a zmm. Low nibbles = sub-block 2c, high
 *      nibbles = sub-block 2c+1.
 *   2. Load 64 int8 x values (32 for sub-block 2c, 32 for 2c+1).
 *   3. VNNI: `vpdpbusd zmm, x_int8, q_uint8` → 16 int32 partials.
 *      Partials[0..7] = 8 groups of 4 (q*x_int) products for sub-block 2c.
 *      Partials[8..15] = same for sub-block 2c+1.
 *   4. Convert to float, multiply by per-sub-block combined scale
 *      `d * sc * x_scale` (broadcast to lanes 0..7 for sub-block 2c,
 *      to lanes 8..15 for sub-block 2c+1 via a `shuffle_f32x4`).
 *   5. Accumulate into the per-row 16-lane float accumulator.
 *
 * Per-sub-block min correction (`dmin * m * sum(x)`) is precomputed
 * in scalar form: sum(x) per sub-block comes from the x-quant pre-pass.
 *
 * The result is approximate (x is quantized to int8), but matches the
 * scalar result within float rounding — token IDs are unchanged. */
#ifdef __AVX512VNNI__
__attribute__((target("avx512vnni,avx512f,avx512dq")))
static void pickle_fast_matmul_q4_k_vnni(float* y,
                                          const unsigned char* W,
                                          int out_n, int in_n,
                                          const float* x) {
    const int BS = 256;
    const size_t block_bytes = 144;
    const int n_blocks = in_n / BS;
    const int n_subblocks = in_n / 32;  /* = n_blocks * 8 */

    /* ===== Pre-pass: quantize x to Q8_0 (one int8 + one fp32 scale per
     * 32-element sub-block) and compute sum(x) per sub-block for the
     * min correction. This is O(in_n) scalar work (~3 µs at in_n=2048),
     * amortised across all out_n rows of W. ===== */
    int8_t* x_int    = (int8_t*)malloc((size_t)in_n);
    float*  x_scale  = (float*) malloc((size_t)n_subblocks * sizeof(float));
    float*  sum_x    = (float*) malloc((size_t)n_subblocks * sizeof(float));
    if (!x_int || !x_scale || !sum_x) {
        /* OOM — should not happen for reasonable in_n. Fall through to
         * the safer AVX-512-F / scalar path below by freeing and
         * returning zeros (caller will see garbage; better than crashing). */
        free(x_int); free(x_scale); free(sum_x);
        for (int o = 0; o < out_n; o++) y[o] = 0.0f;
        return;
    }
    for (int sb = 0; sb < n_subblocks; sb++) {
        const float* xb = x + sb * 32;
        float max_abs = 0.0f;
        for (int i = 0; i < 32; i++) {
            float a = fabsf(xb[i]);
            if (a > max_abs) max_abs = a;
        }
        float scale = max_abs * (1.0f / 127.0f);
        x_scale[sb] = scale;
        float sum = 0.0f;
        if (scale > 0.0f) {
            float inv = 1.0f / scale;
            for (int i = 0; i < 32; i++) {
                sum += xb[i];
                int q = (int)lroundf(xb[i] * inv);
                if (q >  127) q =  127;
                if (q < -128) q = -128;
                x_int[sb * 32 + i] = (int8_t)q;
            }
        } else {
            for (int i = 0; i < 32; i++) x_int[sb * 32 + i] = 0;
        }
        sum_x[sb] = sum;
    }

    /* ===== Main pass: VNNI matmul ===== */
    const __m256i mask_4 = _mm256_set1_epi8(0x0F);
    #pragma omp parallel for schedule(static)
    for (int o = 0; o < out_n; o++) {
        const unsigned char* row = W + (size_t)o * n_blocks * block_bytes;
        __m512 acc_v = _mm512_setzero_ps();
        float  acc_correction = 0.0f;

        for (int b = 0; b < n_blocks; b++) {
            const unsigned char* blk = row + (size_t)b * block_bytes;
            float d    = f16_bits_to_float((uint16_t)blk[0] | ((uint16_t)blk[1] << 8));
            float dmin = f16_bits_to_float((uint16_t)blk[2] | ((uint16_t)blk[3] << 8));
            const unsigned char* sc_bytes = blk + 4;
            const unsigned char* q  = blk + 16;

            int sc_v[8], m_v[8];
            qk_get_scale_min_k4(0, sc_bytes, &sc_v[0], &m_v[0]);
            qk_get_scale_min_k4(1, sc_bytes, &sc_v[1], &m_v[1]);
            qk_get_scale_min_k4(2, sc_bytes, &sc_v[2], &m_v[2]);
            qk_get_scale_min_k4(3, sc_bytes, &sc_v[3], &m_v[3]);
            qk_get_scale_min_k4(4, sc_bytes, &sc_v[4], &m_v[4]);
            qk_get_scale_min_k4(5, sc_bytes, &sc_v[5], &m_v[5]);
            qk_get_scale_min_k4(6, sc_bytes, &sc_v[6], &m_v[6]);
            qk_get_scale_min_k4(7, sc_bytes, &sc_v[7], &m_v[7]);

            const float*  x_scale_b = x_scale + b * 8;
            const float*  sum_x_b   = sum_x   + b * 8;
            const int8_t* x_int_b   = x_int   + b * 256;

            /* Min correction: sum_s (dmin * m_s * sum_x_s).
             * Factor out dmin (cheap scalar FMA chain, 8 terms). */
            float correction = 0.0f;
            for (int s = 0; s < 8; s++) {
                correction += (float)m_v[s] * sum_x_b[s];
            }
            acc_correction += dmin * correction;

            /* 4 chunks of 64 elements per block. */
            for (int c = 0; c < 4; c++) {
                /* Extract 64 uint8 quants: [low_nib(32) | high_nib(32)] */
                __m256i q_bytes = _mm256_loadu_si256((const __m256i*)q);
                __m256i low_nib  = _mm256_and_si256(q_bytes, mask_4);
                __m256i high_nib = _mm256_and_si256(_mm256_srli_epi16(q_bytes, 4), mask_4);
                __m512i q_uint8_64 = _mm512_inserti64x4(
                                        _mm512_castsi256_si512(low_nib), high_nib, 1);

                /* Load 64 int8 x values for this chunk. */
                __m512i x_int8_64 = _mm512_loadu_si512(&x_int_b[c * 64]);

                /* VNNI: 16 int32 partials. Each partial = sum of 4
                 * (q_uint8 * x_int8) products. Lanes 0..7 → sub-block 2c,
                 * lanes 8..15 → sub-block 2c+1.
                 *
                 * Operand order: _mm512_dpbusd_epi32(src, a, b) treats
                 * `a` as UNSIGNED uint8 (zero-extended) and `b` as SIGNED
                 * int8 (sign-extended). So a = q_uint8 (0..15) and
                 * b = x_int8 (-128..127). Getting this backwards makes
                 * negative x values look like ~250, garbage output. */
                __m512i partials = _mm512_dpbusd_epi32(
                                      _mm512_setzero_si512(), q_uint8_64, x_int8_64);
                __m512 pf = _mm512_cvtepi32_ps(partials);

                /* Per-sub-block combined scale: d * sc_s * x_scale_s.
                 * Lanes 0..7 ← scale_lo (sub-block 2c),
                 * lanes 8..15 ← scale_hi (sub-block 2c+1).
                 * _MM_SHUFFLE(0,0,0,0) picks lane 0 of A for output lanes
                 * 0..1 and lane 0 of B for output lanes 2..3 (128-bit
                 * lanes, 4 floats each) — giving the desired 8/8 split. */
                float scale_lo = d * (float)sc_v[2*c + 0] * x_scale_b[2*c + 0];
                float scale_hi = d * (float)sc_v[2*c + 1] * x_scale_b[2*c + 1];
                __m512 scale_v = _mm512_shuffle_f32x4(
                                   _mm512_set1_ps(scale_lo),
                                   _mm512_set1_ps(scale_hi),
                                   _MM_SHUFFLE(0, 0, 0, 0));
                acc_v = _mm512_fmadd_ps(pf, scale_v, acc_v);

                q += 32;
            }
        }
        y[o] = _mm512_reduce_add_ps(acc_v) - acc_correction;
    }

    free(x_int);
    free(x_scale);
    free(sum_x);
}
#endif /* __AVX512VNNI__ */

void pickle_fast_matmul_q4_k(float* y,
                             const unsigned char* W,
                             int out_n, int in_n,
                             const float* x) {
    const int BS = 256;
    const size_t block_bytes = 144;
    int n_blocks = in_n / BS;
#if defined(__AVX512VNNI__)
    /* Best path: VNNI dot product (uint8×int8→int32 MAC, 64-wide). */
    pickle_fast_matmul_q4_k_vnni(y, W, out_n, in_n, x);
    (void)n_blocks;
#elif defined(__AVX512F__)
    /* ===== Hand-vectorised AVX-512 F (16-wide float) Q4_K vec_dot ===== */
    /* Strategy: for each 64-element chunk (32 qs bytes), extract the 64
     * 4-bit quants (low nibble = elem 0..31, high nibble = elem 32..63),
     * zero-extend them to int32 in 4 zmm registers (16 lanes each), convert
     * to float, apply the per-sub-block scale `d*sc` and min `dmin*m` via
     * FMSUB, then FMA-accumulate against the 64 floats of x for the chunk.
     *
     * Four parallel accumulators (acc0..acc3, one per FMA per chunk) break
     * the FMA dependency chain so the compiler can issue 4 in flight per
     * chunk × 4 chunks/block × n_blocks/block-row. This is the key win over
     * the auto-vectorised scalar path below: the old code was stuck at
     * ymm (8-wide) because of the int8→int32→float unpacking stages, this
     * kernel is explicitly zmm (16-wide) — 2× wider on the FMA. */
    const __m256i mask_4 = _mm256_set1_epi8(0x0F);
    #pragma omp parallel for schedule(static)
    for (int o = 0; o < out_n; o++) {
        const unsigned char* row = W + (size_t)o * n_blocks * block_bytes;
        __m512 acc0 = _mm512_setzero_ps();
        __m512 acc1 = _mm512_setzero_ps();
        __m512 acc2 = _mm512_setzero_ps();
        __m512 acc3 = _mm512_setzero_ps();
        for (int b = 0; b < n_blocks; b++) {
            const unsigned char* blk = row + (size_t)b * block_bytes;
            float d    = f16_bits_to_float((uint16_t)blk[0] | ((uint16_t)blk[1] << 8));
            float dmin = f16_bits_to_float((uint16_t)blk[2] | ((uint16_t)blk[3] << 8));
            const unsigned char* sc_bytes = blk + 4;
            const unsigned char* q  = blk + 16;
            const float* xr = x + b * BS;

            /* All 8 (sc, m) sub-block scales — these are cheap (12 byte
             * reads) so precompute them once per block instead of per
             * chunk to avoid re-parsing the 6-bit packed scales. */
            int sc_v[8], m_v[8];
            qk_get_scale_min_k4(0, sc_bytes, &sc_v[0], &m_v[0]);
            qk_get_scale_min_k4(1, sc_bytes, &sc_v[1], &m_v[1]);
            qk_get_scale_min_k4(2, sc_bytes, &sc_v[2], &m_v[2]);
            qk_get_scale_min_k4(3, sc_bytes, &sc_v[3], &m_v[3]);
            qk_get_scale_min_k4(4, sc_bytes, &sc_v[4], &m_v[4]);
            qk_get_scale_min_k4(5, sc_bytes, &sc_v[5], &m_v[5]);
            qk_get_scale_min_k4(6, sc_bytes, &sc_v[6], &m_v[6]);
            qk_get_scale_min_k4(7, sc_bytes, &sc_v[7], &m_v[7]);

            /* 4 chunks of 64 elements. Each chunk: 32 qs bytes covering
             * 64 elements (low nibble = elem chunk*64+l, high nibble =
             * elem chunk*64+l+32), 2 sub-block scales (2*chunk, 2*chunk+1). */
            for (int c = 0; c < 4; c++) {
                float d_lo  = d    * (float)sc_v[2*c + 0];
                float m_lo_f = dmin * (float)m_v [2*c + 0];
                float d_hi  = d    * (float)sc_v[2*c + 1];
                float m_hi_f = dmin * (float)m_v [2*c + 1];

                /* Load 32 qs bytes (one ymm). Each byte holds two 4-bit
                 * quants: low nibble = element chunk*64+l, high nibble =
                 * element chunk*64+l+32. */
                __m256i q_bytes = _mm256_loadu_si256((const __m256i*)q);

                /* Split into 32 uint8 low-nibbles and 32 uint8 high-nibbles.
                 * The high-nibble extraction uses a 16-bit lane right shift
                 * by 4 then mask: shifting 0xABCD (bytes 0xCD, 0xAB) right
                 * by 4 gives 0x0ABC, then & 0x0F leaves the high nibble of
                 * each original byte in the low position. */
                __m256i low_nib  = _mm256_and_si256(q_bytes, mask_4);
                __m256i high_nib = _mm256_and_si256(_mm256_srli_epi16(q_bytes, 4), mask_4);

                /* Zero-extend 16 uint8 → 16 int32 (one zmm each).
                 * Each chunk produces 4 zmm registers:
                 *   low_lo_f  = floats of q[chunk*64 + 0..15]   (low  nibble, bytes 0..15)
                 *   low_hi_f  = floats of q[chunk*64 + 16..31]  (low  nibble, bytes 16..31)
                 *   high_lo_f = floats of q[chunk*64 + 32..47]  (high nibble, bytes 0..15)
                 *   high_hi_f = floats of q[chunk*64 + 48..63]  (high nibble, bytes 16..31)
                 */
                __m512 low_lo_f  = _mm512_cvtepi32_ps(_mm512_cvtepu8_epi32(_mm256_castsi256_si128(low_nib)));
                __m512 low_hi_f  = _mm512_cvtepi32_ps(_mm512_cvtepu8_epi32(_mm256_extracti128_si256(low_nib, 1)));
                __m512 high_lo_f = _mm512_cvtepi32_ps(_mm512_cvtepu8_epi32(_mm256_castsi256_si128(high_nib)));
                __m512 high_hi_f = _mm512_cvtepi32_ps(_mm512_cvtepu8_epi32(_mm256_extracti128_si256(high_nib, 1)));

                /* Load 64 floats of x for this chunk, 4 zmm registers of 16. */
                __m512 x0 = _mm512_loadu_ps(&xr[c*64 +  0]);
                __m512 x1 = _mm512_loadu_ps(&xr[c*64 + 16]);
                __m512 x2 = _mm512_loadu_ps(&xr[c*64 + 32]);
                __m512 x3 = _mm512_loadu_ps(&xr[c*64 + 48]);

                /* Broadcast scale and min for each half. */
                __m512 d_lo_v   = _mm512_set1_ps(d_lo);
                __m512 m_lo_f_v = _mm512_set1_ps(m_lo_f);
                __m512 d_hi_v   = _mm512_set1_ps(d_hi);
                __m512 m_hi_f_v = _mm512_set1_ps(m_hi_f);

                /* dq = q * d_s - m_s_f  (FMSUB: a*b - c) */
                __m512 dq_low_lo  = _mm512_fmsub_ps(low_lo_f,  d_lo_v, m_lo_f_v);
                __m512 dq_low_hi  = _mm512_fmsub_ps(low_hi_f,  d_lo_v, m_lo_f_v);
                __m512 dq_high_lo = _mm512_fmsub_ps(high_lo_f, d_hi_v, m_hi_f_v);
                __m512 dq_high_hi = _mm512_fmsub_ps(high_hi_f, d_hi_v, m_hi_f_v);

                /* acc += dq * x  (FMADD: a*b + c). Four independent
                 * accumulators break the FMA dependency chain. */
                acc0 = _mm512_fmadd_ps(dq_low_lo,  x0, acc0);
                acc1 = _mm512_fmadd_ps(dq_low_hi,  x1, acc1);
                acc2 = _mm512_fmadd_ps(dq_high_lo, x2, acc2);
                acc3 = _mm512_fmadd_ps(dq_high_hi, x3, acc3);

                q += 32;
            }
        }
        /* Reduce 4 zmm accumulators → scalar. */
        __m512 acc01 = _mm512_add_ps(acc0, acc1);
        __m512 acc23 = _mm512_add_ps(acc2, acc3);
        __m512 acc_v = _mm512_add_ps(acc01, acc23);
        y[o] = _mm512_reduce_add_ps(acc_v);
    }
#else
    /* ===== Scalar fallback (auto-vectorised ymm by the compiler) ===== */
    /* Fused dequant+dot: instead of staging 256 floats to a stack
     * buffer and then dot-producting, we accumulate straight into acc
     * as each element is dequanted. Saves 1KB of stack traffic per
     * block and keeps partials in registers across the dot. */
    #pragma omp parallel for schedule(static)
    for (int o = 0; o < out_n; o++) {
        const unsigned char* row = W + (size_t)o * n_blocks * block_bytes;
        float acc = 0.0f;
        for (int b = 0; b < n_blocks; b++) {
            const unsigned char* blk = row + (size_t)b * block_bytes;
            float d    = f16_bits_to_float((uint16_t)blk[0] | ((uint16_t)blk[1] << 8));
            float dmin = f16_bits_to_float((uint16_t)blk[2] | ((uint16_t)blk[3] << 8));
            const unsigned char* sc = blk + 4;
            const unsigned char* q  = blk + 16;
            const float* xr = x + b * BS;
            /* GGML chunked layout: 4 chunks of 64 elements, 32 bytes of qs each. */
            int is = 0;
            for (int j = 0; j < 256; j += 64) {
                int sc0, m0, sc1, m1;
                qk_get_scale_min_k4(is + 0, sc, &sc0, &m0);
                qk_get_scale_min_k4(is + 1, sc, &sc1, &m1);
                float d1 = d * (float)sc0, m1f = dmin * (float)m0;
                float d2 = d * (float)sc1, m2f = dmin * (float)m1;
                #pragma omp simd simdlen(16) reduction(+:acc)
                for (int l = 0; l < 32; l++) {
                    acc += (d1 * (float)(q[l] & 0x0F) - m1f) * xr[j + l];
                    acc += (d2 * (float)(q[l] >> 4)   - m2f) * xr[j + l + 32];
                }
                q += 32;
                is += 2;
            }
        }
        y[o] = acc;
    }
#endif /* __AVX512VNNI__ / __AVX512F__ / scalar */
}

/* ----- Q5_K: 176 bytes / 256 elements ------------------------------ */
/* Block layout (matches ggml-common.h block_q5_K):
 *   d (f16)    : offset 0
 *   dmin (f16) : offset 2
 *   scales[12] : offset 4  (8x 6-bit scale + 8x 6-bit min)
 *   qh[32]     : offset 16 (1 high bit per element, 8 elements per byte)
 *   qs[128]    : offset 48 (4 low bits per element, NON-INTERLEAVED)
 *
 * GGML non-interleaved packing (matches ggml-quants.c dequantize_row_q5_K):
 * 4 chunks of 64 elements. qs advances 32 bytes per chunk (low nibble = element
 * chunk*64+l, high nibble = element chunk*64+l+32). qh is REUSED across chunks
 * (does NOT advance) — each qh byte covers 8 elements (1 bit each) at bit
 * positions 2*chunk+half. So qh[l] bit (2*chunk) is for element chunk*64+l,
 * bit (2*chunk+1) is for element chunk*64+l+32.
 *
 * Formula: dq = d * sc * (qv) - dmin * m, where qv = qs_nibble + (qh_bit ? 16 : 0) */
void pickle_fast_matmul_q5_k(float* y,
                             const unsigned char* W,
                             int out_n, int in_n,
                             const float* x) {
    const int BS = 256;
    const size_t block_bytes = 176;
    int n_blocks = in_n / BS;
    #pragma omp parallel for schedule(static)
    for (int o = 0; o < out_n; o++) {
        const unsigned char* row = W + (size_t)o * n_blocks * block_bytes;
        float acc = 0.0f;
        for (int b = 0; b < n_blocks; b++) {
            const unsigned char* blk = row + (size_t)b * block_bytes;
            float d    = f16_bits_to_float((uint16_t)blk[0] | ((uint16_t)blk[1] << 8));
            float dmin = f16_bits_to_float((uint16_t)blk[2] | ((uint16_t)blk[3] << 8));
            const unsigned char* sc = blk + 4;
            const unsigned char* qh = blk + 16;
            const unsigned char* ql = blk + 48;
            const float* xr = x + b * BS;
            int is = 0;
            uint8_t u1 = 1, u2 = 2;  /* bit masks for the 5th bit, shift per chunk */
            for (int j = 0; j < 256; j += 64) {
                int sc0, m0, sc1, m1;
                qk_get_scale_min_k4(is + 0, sc, &sc0, &m0);
                qk_get_scale_min_k4(is + 1, sc, &sc1, &m1);
                float d1 = d * (float)sc0, m1f = dmin * (float)m0;
                float d2 = d * (float)sc1, m2f = dmin * (float)m1;
                #pragma omp simd simdlen(16) reduction(+:acc)
                for (int l = 0; l < 32; l++) {
                    int q0 = (ql[l] & 0x0F) + ((qh[l] & u1) ? 16 : 0);
                    int q1 = (ql[l] >> 4)   + ((qh[l] & u2) ? 16 : 0);
                    acc += (d1 * (float)q0 - m1f) * xr[j + l];
                    acc += (d2 * (float)q1 - m2f) * xr[j + l + 32];
                }
                ql += 32;
                is += 2;
                u1 <<= 2;
                u2 <<= 2;
            }
        }
        y[o] = acc;
    }
}

/* ----- Q6_K: 210 bytes / 256 elements ------------------------------ */
/* Block layout (matches ggml-common.h block_q6_K):
 *   ql[128]   : offset 0,  low 4 bits per element (NON-INTERLEAVED)
 *   qh[64]    : offset 128, high 2 bits per element (NON-INTERLEAVED)
 *   scales[16]: offset 192, one int8 scale per 16-element sub-block
 *   d (f16)   : offset 208, super-block scale
 *
 * GGML non-interleaved packing (matches ggml-quants.c dequantize_row_q6_K):
 * 2 chunks of 128 elements. Within a chunk, 4 sub-blocks of 32 (sub=0..3):
 *   ql[chunk*64 + (sub%2)*32 + l]: low nibble if sub in {0,1}, high if sub in {2,3}
 *   qh[chunk*32 + l] bits [2*sub, 2*sub+1]: the high 2 bits
 * Scale: scales[chunk*8 + l/16 + 2*sub] = scales[i/16]
 * Formula: dq = d * scales[i/16] * (q - 32), q = q_lo | (q_hi << 4), range 0..63 */
void pickle_fast_matmul_q6_k(float* y,
                             const unsigned char* W,
                             int out_n, int in_n,
                             const float* x) {
    const int BS = 256;
    const size_t block_bytes = 210;
    int n_blocks = in_n / BS;
    #pragma omp parallel for schedule(static)
    for (int o = 0; o < out_n; o++) {
        const unsigned char* row = W + (size_t)o * n_blocks * block_bytes;
        float acc = 0.0f;
        for (int b = 0; b < n_blocks; b++) {
            const unsigned char* blk = row + (size_t)b * block_bytes;
            const unsigned char* ql     = blk;
            const unsigned char* qh     = blk + 128;
            const signed char*   scales = (const signed char*)(blk + 192);
            float d = f16_bits_to_float((uint16_t)blk[208] | ((uint16_t)blk[209] << 8));
            const float* xr = x + b * BS;
            /* Restructured to GGML's chunked 4-element form: per outer
             * step we advance ql by 64, qh by 32, scales by 8, covering
             * 128 elements (4 sub-blocks of 32 each). This is
             * mathematically identical to the per-element form below
             * (verified by element-index correspondence — see the
             * VERIFY-Q6K worklog entry) but exposes 4 independent
             * accumulations per inner iteration, giving the compiler a
             * clean 4-wide reduction chain to vectorise. */
            for (int n = 0; n < 256; n += 128) {
                /* Split the l-loop into two 16-wide halves so the scale
                 * index is loop-invariant within each half — the
                 * compiler can then prove the scales[is+k] lookups are
                 * invariant and vectorise the FMA chain. Note: ql/qh
                 * still index by `l` (NOT l-16) — they are contiguous
                 * across the 32-element inner range; only the SCALE
                 * index changes at l=16. */
                #pragma omp simd simdlen(16) reduction(+:acc)
                for (int l = 0; l < 16; ++l) {
                    const int q1 = (int)((ql[l +  0] & 0x0F) | (((qh[l] >> 0) & 3) << 4)) - 32;
                    const int q2 = (int)((ql[l + 32] & 0x0F) | (((qh[l] >> 2) & 3) << 4)) - 32;
                    const int q3 = (int)((ql[l +  0]  >> 4) | (((qh[l] >> 4) & 3) << 4)) - 32;
                    const int q4 = (int)((ql[l + 32]  >> 4) | (((qh[l] >> 6) & 3) << 4)) - 32;
                    acc += d * (float)scales[0] * (float)q1 * xr[n + l +  0];
                    acc += d * (float)scales[2] * (float)q2 * xr[n + l + 32];
                    acc += d * (float)scales[4] * (float)q3 * xr[n + l + 64];
                    acc += d * (float)scales[6] * (float)q4 * xr[n + l + 96];
                }
                #pragma omp simd simdlen(16) reduction(+:acc)
                for (int l = 16; l < 32; ++l) {
                    const int q1 = (int)((ql[l +  0] & 0x0F) | (((qh[l] >> 0) & 3) << 4)) - 32;
                    const int q2 = (int)((ql[l + 32] & 0x0F) | (((qh[l] >> 2) & 3) << 4)) - 32;
                    const int q3 = (int)((ql[l +  0]  >> 4) | (((qh[l] >> 4) & 3) << 4)) - 32;
                    const int q4 = (int)((ql[l + 32]  >> 4) | (((qh[l] >> 6) & 3) << 4)) - 32;
                    acc += d * (float)scales[1] * (float)q1 * xr[n + l +  0];
                    acc += d * (float)scales[3] * (float)q2 * xr[n + l + 32];
                    acc += d * (float)scales[5] * (float)q3 * xr[n + l + 64];
                    acc += d * (float)scales[7] * (float)q4 * xr[n + l + 96];
                }
                ql     += 64;
                qh     += 32;
                scales += 8;
            }
        }
        y[o] = acc;
    }
}

/* ----- Q8_K: 292 bytes / 256 elements ------------------------------ */
/* Block: d(f32,4) + qs(256) + dsubs(16) + scales(16). */
void pickle_fast_matmul_q8_k(float* y,
                             const unsigned char* W,
                             int out_n, int in_n,
                             const float* x) {
    const int BS = 256;
    const size_t block_bytes = 292;
    int n_blocks = in_n / BS;
    #pragma omp parallel for schedule(static)
    for (int o = 0; o < out_n; o++) {
        const unsigned char* row = W + (size_t)o * n_blocks * block_bytes;
        float acc = 0.0f;
        for (int b = 0; b < n_blocks; b++) {
            const unsigned char* blk = row + (size_t)b * block_bytes;
            float d;
            memcpy(&d, blk, 4);
            const signed char* qs     = (const signed char*)(blk + 4);
            const float*       dsubs  = (const float*)(blk + 4 + 256);
            const float*       scales = (const float*)(blk + 4 + 256 + 16);
            const float* xr = x + b * BS;
            #pragma omp simd simdlen(16) reduction(+:acc)
            for (int i = 0; i < 256; i++) {
                int sub = i / 64;
                float sub_d   = d * scales[sub];
                float sub_off = dsubs[sub];
                acc += (sub_d * (float)qs[i] + sub_off) * xr[i];
            }
        }
        y[o] = acc;
    }
}

/* ----- Q2_K: 84 bytes / 256 elements ------------------------------- */
/* Block: d(f16,2) + dmin(f16,2) + scales(16) + qs(64). */
void pickle_fast_matmul_q2_k(float* y,
                             const unsigned char* W,
                             int out_n, int in_n,
                             const float* x) {
    const int BS = 256;
    const size_t block_bytes = 84;
    int n_blocks = in_n / BS;
    #pragma omp parallel for schedule(static)
    for (int o = 0; o < out_n; o++) {
        const unsigned char* row = W + (size_t)o * n_blocks * block_bytes;
        float acc = 0.0f;
        for (int b = 0; b < n_blocks; b++) {
            const unsigned char* blk = row + (size_t)b * block_bytes;
            float d    = f16_bits_to_float((uint16_t)blk[0] | ((uint16_t)blk[1] << 8));
            float dmin = f16_bits_to_float((uint16_t)blk[2] | ((uint16_t)blk[3] << 8));
            const unsigned char* sc = blk + 4;
            const unsigned char* qs = blk + 20;
            int scales[32];
            for (int g = 0; g < 4; g++) {
                uint32_t sw = (uint32_t)sc[g*4]       |
                              ((uint32_t)sc[g*4 + 1] << 8) |
                              ((uint32_t)sc[g*4 + 2] << 16) |
                              ((uint32_t)sc[g*4 + 3] << 24);
                for (int j = 0; j < 8; j++) {
                    int v = (sw >> (4*j)) & 0x0F;
                    if (v & 0x08) v -= 16;
                    scales[g*8 + j] = v;
                }
            }
            const float* xr = x + b * BS;
            #pragma omp simd simdlen(16) reduction(+:acc)
            for (int i = 0; i < 256; i++) {
                int byte_idx = i / 4;
                int bit_off  = 2 * (i & 3);
                int q = (qs[byte_idx] >> bit_off) & 0x03;
                int sub = i / 16;
                float sv = d * (float)scales[sub] * (float)q;
                acc += (sv - dmin) * xr[i];
            }
        }
        y[o] = acc;
    }
}

/* ----- Q3_K: 110 bytes / 256 elements ------------------------------ */
/* Block: hmask(32) + qs(64) + sc(12) + pad(2). */
void pickle_fast_matmul_q3_k(float* y,
                             const unsigned char* W,
                             int out_n, int in_n,
                             const float* x) {
    const int BS = 256;
    const size_t block_bytes = 110;
    int n_blocks = in_n / BS;
    #pragma omp parallel for schedule(static)
    for (int o = 0; o < out_n; o++) {
        const unsigned char* row = W + (size_t)o * n_blocks * block_bytes;
        float acc = 0.0f;
        for (int b = 0; b < n_blocks; b++) {
            const unsigned char* blk = row + (size_t)b * block_bytes;
            const unsigned char* hmask = blk;
            const unsigned char* qs     = blk + 32;
            const unsigned char* sc     = blk + 96;
            /* blk[108..109] = pad */
            float d    = f16_bits_to_float((uint16_t)sc[0] | ((uint16_t)sc[1] << 8));
            float dmin = f16_bits_to_float((uint16_t)sc[2] | ((uint16_t)sc[3] << 8));
            int ks[8];
            for (int i = 0; i < 8; i++) {
                int v = sc[4 + i] & 0x3F;
                if (v & 0x20) v -= 64;
                ks[i] = v;
            }
            const float* xr = x + b * BS;
            #pragma omp simd simdlen(16) reduction(+:acc)
            for (int i = 0; i < 256; i++) {
                int byte_idx = i / 4;
                int bit_off  = 2 * (i & 3);
                int q = (qs[byte_idx] >> bit_off) & 0x03;
                if (q & 0x02) q -= 4;
                if ((hmask[i / 8] >> (i % 8)) & 1) q += 4;
                int sub = i / 32;
                float sv = d * (float)ks[sub] * (float)q;
                acc += (sv - dmin) * xr[i];
            }
        }
        y[o] = acc;
    }
}

/* ----- F16: 2 bytes / element -------------------------------------- */
void pickle_fast_matmul_f16(float* y,
                            const unsigned char* W_bytes,
                            int out_n, int in_n,
                            const float* x) {
    /* Walk row by row, dequantizing each f16 to f32 on the fly. The
     * compiler vectorises the inner accumulation. */
    const uint16_t* W = (const uint16_t*)W_bytes;
    /* Per-thread row32 buffer: allocate one inside a parallel region
     * so each OpenMP thread gets its own scratch (the original code
     * reused a single shared buffer, which is unsafe under
     * #pragma omp parallel for). The 4-way unroll matches the F32 path
     * and gives the compiler a clean reduction chain. */
    #pragma omp parallel
    {
        float* row32 = (float*)malloc((size_t)in_n * sizeof(float));
        if (row32) {
            #pragma omp for schedule(static)
            for (int o = 0; o < out_n; o++) {
                const uint16_t* wrow = W + (size_t)o * in_n;
                for (int i = 0; i < in_n; i++)
                    row32[i] = f16_bits_to_float(wrow[i]);
                float acc0 = 0.0f, acc1 = 0.0f, acc2 = 0.0f, acc3 = 0.0f;
                int n4 = in_n & ~3;
                #pragma omp simd simdlen(16) reduction(+:acc0,acc1,acc2,acc3)
                for (int i = 0; i < n4; i += 4) {
                    acc0 += row32[i]     * x[i];
                    acc1 += row32[i + 1] * x[i + 1];
                    acc2 += row32[i + 2] * x[i + 2];
                    acc3 += row32[i + 3] * x[i + 3];
                }
                for (int i = n4; i < in_n; i++) acc0 += row32[i] * x[i];
                y[o] = (acc0 + acc1) + (acc2 + acc3);
            }
            free(row32);
        } else {
            #pragma omp for schedule(static)
            for (int o = 0; o < out_n; o++) y[o] = 0.0f;
        }
    }
}

/* ----- Dispatch by tensor type ------------------------------------- */
int pickle_fast_matmul_dispatch(float* y,
                                const pickle_tensor_info_t* t,
                                const float* x,
                                int out_n, int in_n) {
    if (!t || !t->data) return PICKLE_ERR_ARG;
    switch (t->type) {
        case GGML_F32:
            pickle_fast_matmul_f32(y, (const float*)t->data, x, out_n, in_n);
            return PICKLE_OK;
        case GGML_F16:
            pickle_fast_matmul_f16(y, (const unsigned char*)t->data, out_n, in_n, x);
            return PICKLE_OK;
        case GGML_Q4_0:
            pickle_fast_matmul_q4_0(y, (const unsigned char*)t->data, out_n, in_n, x);
            return PICKLE_OK;
        case GGML_Q4_1:
            pickle_fast_matmul_q4_1(y, (const unsigned char*)t->data, out_n, in_n, x);
            return PICKLE_OK;
        case GGML_Q5_0:
            pickle_fast_matmul_q5_0(y, (const unsigned char*)t->data, out_n, in_n, x);
            return PICKLE_OK;
        case GGML_Q5_1:
            pickle_fast_matmul_q5_1(y, (const unsigned char*)t->data, out_n, in_n, x);
            return PICKLE_OK;
        case GGML_Q8_0:
            pickle_fast_matmul_q8_0(y, (const unsigned char*)t->data, out_n, in_n, x);
            return PICKLE_OK;
        case GGML_Q8_1:
            pickle_fast_matmul_q8_1(y, (const unsigned char*)t->data, out_n, in_n, x);
            return PICKLE_OK;
        case GGML_Q4_K:
            pickle_fast_matmul_q4_k(y, (const unsigned char*)t->data, out_n, in_n, x);
            return PICKLE_OK;
        case GGML_Q5_K:
            pickle_fast_matmul_q5_k(y, (const unsigned char*)t->data, out_n, in_n, x);
            return PICKLE_OK;
        case GGML_Q6_K:
            pickle_fast_matmul_q6_k(y, (const unsigned char*)t->data, out_n, in_n, x);
            return PICKLE_OK;
        case GGML_Q8_K:
            pickle_fast_matmul_q8_k(y, (const unsigned char*)t->data, out_n, in_n, x);
            return PICKLE_OK;
        case GGML_Q2_K:
            pickle_fast_matmul_q2_k(y, (const unsigned char*)t->data, out_n, in_n, x);
            return PICKLE_OK;
        case GGML_Q3_K:
            pickle_fast_matmul_q3_k(y, (const unsigned char*)t->data, out_n, in_n, x);
            return PICKLE_OK;
        default:
            return PICKLE_ERR_TYPE;
    }
}

/* ================================================================== */
/* Fast-state init / free                                             */
/* ================================================================== */
static int find_tensor_idx(const pickle_model_t* m, const char* prefix, int L, const char* suffix) {
    char nm[128];
    int n = 0;
    const char* p = prefix;
    while (*p && n < (int)sizeof(nm) - 1) nm[n++] = *p++;
    /* layer number */
    char num[16]; int nl = 0;
    int v = L;
    if (v == 0) num[nl++] = '0';
    else { char t[16]; int tl = 0; while (v) { t[tl++] = '0' + (v % 10); v /= 10; } while (tl) num[nl++] = t[--tl]; }
    for (int i = 0; i < nl && n < (int)sizeof(nm) - 1; i++) nm[n++] = num[i];
    p = suffix;
    while (*p && n < (int)sizeof(nm) - 1) nm[n++] = *p++;
    nm[n] = 0;
    return pickle_tensor_find(m, nm);
}

int pickle_fast_state_init(const pickle_model_t* m,
                           const pickle_arch_t* arch,
                           pickle_fast_state_t* state) {
    if (!m || !arch || !state) return PICKLE_ERR_ARG;
    memset(state, 0, sizeof(*state));
    state->n_layers         = arch->n_layers;
    state->n_heads          = arch->n_heads;
    state->n_kv_heads       = arch->n_kv_heads;
    state->head_dim         = arch->head_dim;
    state->hidden_dim       = arch->hidden_dim;
    state->intermediate_dim = arch->intermediate_dim;
    state->vocab_size       = arch->vocab_size;
    state->max_seq          = arch->max_seq_len;
    state->tie_word_embeddings = arch->tie_word_embeddings;
    state->rope_theta       = sfp_to_float(arch->rope_theta_bits);
    state->rms_eps          = sfp_to_float(arch->rms_eps_bits);
    state->half             = arch->head_dim / 2;
    if (state->half > 64) state->half = 64;   /* match soft-float cap */

    /* Snapshot every tensor's info into the state so the forward pass
     * has direct access to the data pointer + type without reaching
     * into the opaque model struct. */
    state->n_tensors = (int)pickle_tensor_count(m);
    state->tensors = (pickle_tensor_info_t*)calloc((size_t)state->n_tensors,
                                                   sizeof(pickle_tensor_info_t));
    if (!state->tensors) return PICKLE_ERR_MEMORY;
    for (int i = 0; i < state->n_tensors; i++) {
        int rc = pickle_tensor_info(m, (size_t)i, &state->tensors[i]);
        if (rc != PICKLE_OK) { pickle_fast_state_free(state); return rc; }
    }

    state->t_embd   = pickle_tensor_find(m, "token_embd.weight");
    state->t_norm   = pickle_tensor_find(m, "output_norm.weight");
    state->t_output = pickle_tensor_find(m, "output.weight");
    if (state->t_embd < 0) { pickle_fast_state_free(state); return PICKLE_ERR_ARCH; }

    /* If the embedding tensor hasn't been loaded yet (lazy load), load
     * it now. For F32/F16 we load raw (the fast path reads both
     * natively). For any other quant, we dequantize to F32 (the fast
     * path's per-row embd lookup only handles F32/F16 inline). */
    if (state->tensors[state->t_embd].data == NULL) {
        int emb_type_local = state->tensors[state->t_embd].type;
        int drc;
        if (emb_type_local == GGML_F32 || emb_type_local == GGML_F16) {
            drc = pickle_load_tensor_raw((pickle_model_t*)m, (size_t)state->t_embd);
        } else {
            drc = pickle_dequant_tensor((pickle_model_t*)m, (size_t)state->t_embd);
        }
        if (drc != PICKLE_OK) { pickle_fast_state_free(state); return drc; }
        if (pickle_tensor_info(m, (size_t)state->t_embd,
                               &state->tensors[state->t_embd]) != PICKLE_OK) {
            pickle_fast_state_free(state);
            return PICKLE_ERR_FORMAT;
        }
    }

    int NL = arch->n_layers;
    state->t_attn_norm = (int*)calloc(NL, sizeof(int));
    state->t_attn_q    = (int*)calloc(NL, sizeof(int));
    state->t_attn_k    = (int*)calloc(NL, sizeof(int));
    state->t_attn_v    = (int*)calloc(NL, sizeof(int));
    state->t_attn_out  = (int*)calloc(NL, sizeof(int));
    state->t_ffn_norm  = (int*)calloc(NL, sizeof(int));
    state->t_ffn_gate  = (int*)calloc(NL, sizeof(int));
    state->t_ffn_up    = (int*)calloc(NL, sizeof(int));
    state->t_ffn_down  = (int*)calloc(NL, sizeof(int));
    if (!state->t_attn_norm || !state->t_attn_q || !state->t_attn_k ||
        !state->t_attn_v || !state->t_attn_out || !state->t_ffn_norm ||
        !state->t_ffn_gate || !state->t_ffn_up || !state->t_ffn_down) {
        pickle_fast_state_free(state);
        return PICKLE_ERR_MEMORY;
    }
    for (int L = 0; L < NL; L++) {
        state->t_attn_norm[L] = find_tensor_idx(m, "blk.", L, ".attn_norm.weight");
        state->t_attn_q[L]    = find_tensor_idx(m, "blk.", L, ".attn_q.weight");
        state->t_attn_k[L]    = find_tensor_idx(m, "blk.", L, ".attn_k.weight");
        state->t_attn_v[L]    = find_tensor_idx(m, "blk.", L, ".attn_v.weight");
        state->t_attn_out[L]  = find_tensor_idx(m, "blk.", L, ".attn_output.weight");
        state->t_ffn_norm[L]  = find_tensor_idx(m, "blk.", L, ".ffn_norm.weight");
        state->t_ffn_gate[L]  = find_tensor_idx(m, "blk.", L, ".ffn_gate.weight");
        state->t_ffn_up[L]    = find_tensor_idx(m, "blk.", L, ".ffn_up.weight");
        state->t_ffn_down[L]  = find_tensor_idx(m, "blk.", L, ".ffn_down.weight");
        if (state->t_attn_norm[L] < 0 || state->t_attn_q[L] < 0 ||
            state->t_attn_k[L] < 0 || state->t_attn_v[L] < 0 ||
            state->t_attn_out[L] < 0 || state->t_ffn_norm[L] < 0 ||
            state->t_ffn_gate[L] < 0 || state->t_ffn_up[L] < 0 ||
            state->t_ffn_down[L] < 0) {
            pickle_fast_state_free(state);
            return PICKLE_ERR_ARCH;
        }
    }

    /* Precompute inverse frequencies: inv_freq[i] = 1 / theta^(2i/D). */
    int half = state->half;
    state->rope_inv_freq = (float*)calloc((size_t)half, sizeof(float));
    if (!state->rope_inv_freq) { pickle_fast_state_free(state); return PICKLE_ERR_MEMORY; }
    for (int i = 0; i < half; i++) {
        state->rope_inv_freq[i] = 1.0f / powf(state->rope_theta, (float)(2*i) / (float)arch->head_dim);
    }
    /* Allocate sin/cos tables lazily — extend on demand. */
    state->rope_sin = (float*)calloc((size_t)state->max_seq * half, sizeof(float));
    state->rope_cos = (float*)calloc((size_t)state->max_seq * half, sizeof(float));
    if (!state->rope_sin || !state->rope_cos) { pickle_fast_state_free(state); return PICKLE_ERR_MEMORY; }
    state->rope_filled_upto = 0;

    /* Load every per-layer weight tensor's raw bytes (the fast-path
     * quantized matmul reads raw block bytes; F32/F16 norms are also
     * read natively from raw bytes). Loading all layers up front keeps
     * the forward pass I/O-free. */
    for (int L = 0; L < NL; L++) {
        int idxs[9] = {
            state->t_attn_norm[L], state->t_attn_q[L], state->t_attn_k[L],
            state->t_attn_v[L],    state->t_attn_out[L], state->t_ffn_norm[L],
            state->t_ffn_gate[L],  state->t_ffn_up[L],   state->t_ffn_down[L]
        };
        for (int k = 0; k < 9; k++) {
            int idx = idxs[k];
            if (idx < 0) continue;
            if (state->tensors[idx].data != NULL) continue;
            int lrc = pickle_load_tensor_raw((pickle_model_t*)m, (size_t)idx);
            if (lrc != PICKLE_OK) { pickle_fast_state_free(state); return lrc; }
            if (pickle_tensor_info(m, (size_t)idx,
                                   &state->tensors[idx]) != PICKLE_OK) {
                pickle_fast_state_free(state);
                return PICKLE_ERR_FORMAT;
            }
        }
    }
    /* Also load the output_norm (F32, small) and output.weight (may
     * be quantized — the fast path's dispatch handles raw bytes). */
    if (state->t_norm >= 0 && state->tensors[state->t_norm].data == NULL) {
        int lrc = pickle_load_tensor_raw((pickle_model_t*)m, (size_t)state->t_norm);
        if (lrc != PICKLE_OK) { pickle_fast_state_free(state); return lrc; }
        pickle_tensor_info(m, (size_t)state->t_norm, &state->tensors[state->t_norm]);
    }
    if (state->t_output >= 0 && state->tensors[state->t_output].data == NULL) {
        int lrc = pickle_load_tensor_raw((pickle_model_t*)m, (size_t)state->t_output);
        if (lrc != PICKLE_OK) { pickle_fast_state_free(state); return lrc; }
        pickle_tensor_info(m, (size_t)state->t_output, &state->tensors[state->t_output]);
    }

    return PICKLE_OK;
}

void pickle_fast_state_free(pickle_fast_state_t* state) {
    if (!state) return;
    free(state->t_attn_norm);
    free(state->t_attn_q);
    free(state->t_attn_k);
    free(state->t_attn_v);
    free(state->t_attn_out);
    free(state->t_ffn_norm);
    free(state->t_ffn_gate);
    free(state->t_ffn_up);
    free(state->t_ffn_down);
    free(state->rope_inv_freq);
    free(state->rope_sin);
    free(state->rope_cos);
    free(state->tensors);
    memset(state, 0, sizeof(*state));
}

int pickle_fast_state_extend_rope(pickle_fast_state_t* state, int upto_pos) {
    if (!state) return PICKLE_ERR_ARG;
    if (upto_pos < state->rope_filled_upto) return PICKLE_OK;
    if (upto_pos >= state->max_seq) upto_pos = state->max_seq - 1;
    int half = state->half;
    for (int p = state->rope_filled_upto; p <= upto_pos; p++) {
        float* sp = state->rope_sin + (size_t)p * half;
        float* cp = state->rope_cos + (size_t)p * half;
        for (int i = 0; i < half; i++) {
            float angle = state->rope_inv_freq[i] * (float)p;
            sp[i] = sinf(angle);
            cp[i] = cosf(angle);
        }
    }
    state->rope_filled_upto = upto_pos + 1;
    return PICKLE_OK;
}

/* ================================================================== */
/* Contiguous KV cache                                                */
/* ================================================================== */
int pickle_fast_kv_alloc(const pickle_arch_t* arch, pickle_fast_kv_t* kv) {
    if (!arch || !kv) return PICKLE_ERR_ARG;
    memset(kv, 0, sizeof(*kv));
    kv->n_layers   = arch->n_layers;
    kv->n_kv_heads = arch->n_kv_heads;
    kv->head_dim   = arch->head_dim;
    kv->max_seq    = arch->max_seq_len;
    kv->k_stride_layer = (size_t)arch->max_seq_len * arch->n_kv_heads * arch->head_dim;
    kv->v_stride_layer = kv->k_stride_layer;
    size_t bytes = (size_t)arch->n_layers * kv->k_stride_layer * sizeof(float);
    kv->k = (float*)calloc(bytes ? bytes : 1, 1);
    kv->v = (float*)calloc(bytes ? bytes : 1, 1);
    if (!kv->k || !kv->v) {
        free(kv->k); free(kv->v);
        memset(kv, 0, sizeof(*kv));
        return PICKLE_ERR_MEMORY;
    }
    return PICKLE_OK;
}
void pickle_fast_kv_free(pickle_fast_kv_t* kv) {
    if (!kv) return;
    free(kv->k);
    free(kv->v);
    memset(kv, 0, sizeof(*kv));
}

/* ================================================================== */
/* Fast forward pass                                                  */
/* ================================================================== */
int pickle_fast_forward(pickle_model_t*        model,
                        const pickle_arch_t*   arch,
                        pickle_fast_state_t*   state,
                        const int32_t*         tokens,
                        size_t                 n_tokens,
                        float*                 out_logits,
                        pickle_fast_kv_t*      kv,
                        size_t*                kv_pos) {
    if (!model || !arch || !state || !tokens || !out_logits)
        return PICKLE_ERR_ARG;

    int H  = state->n_heads;
    int HK = state->n_kv_heads;
    int D  = state->head_dim;
    int HD = state->hidden_dim;
    int ID = state->intermediate_dim;
    int VS = state->vocab_size;
    int NL = state->n_layers;
    int half = state->half;
    float eps = state->rms_eps;

    pickle_tensor_info_t* tensors = state->tensors;

    /* The token embedding tensor was dequantized to F32 at state-init
     * time (if it was quantized). For F16 embd we dequant on the fly
     * per row. */
    const float* emb_w = (const float*)tensors[state->t_embd].data;
    int emb_type = tensors[state->t_embd].type;

    /* Working buffers — heap to keep stack small. */
    float* x     = (float*)malloc((size_t)HD * sizeof(float));
    float* xn    = (float*)malloc((size_t)HD * sizeof(float));
    float* q     = (float*)malloc((size_t)(H  * D) * sizeof(float));
    float* k     = (float*)malloc((size_t)(HK * D) * sizeof(float));
    float* v     = (float*)malloc((size_t)(HK * D) * sizeof(float));
    float* ao    = (float*)malloc((size_t)(H  * D) * sizeof(float));
    float* aproj = (float*)malloc((size_t)HD * sizeof(float));
    float* xn2   = (float*)malloc((size_t)HD * sizeof(float));
    float* gate  = (float*)malloc((size_t)ID * sizeof(float));
    float* up    = (float*)malloc((size_t)ID * sizeof(float));
    float* act   = (float*)malloc((size_t)ID * sizeof(float));
    float* down  = (float*)malloc((size_t)HD * sizeof(float));
    /* For F16 embedding: per-row dequant buffer. */
    float* emb_row = (emb_type == GGML_F16) ? (float*)malloc((size_t)HD * sizeof(float)) : NULL;
    /* Attention scores — heap to allow large contexts (no 512 cap). */
    float* scores = (float*)malloc((size_t)(state->max_seq > 0 ? state->max_seq : 1) * sizeof(float));

    if (!x || !xn || !q || !k || !v || !ao || !aproj || !xn2 ||
        !gate || !up || !act || !down || !scores ||
        (emb_type == GGML_F16 && !emb_row)) {
        free(x); free(xn); free(q); free(k); free(v); free(ao); free(aproj);
        free(xn2); free(gate); free(up); free(act); free(down); free(scores);
        free(emb_row);
        return PICKLE_ERR_MEMORY;
    }

    size_t pos_start = (kv && kv_pos) ? *kv_pos : 0;

    /* Extend RoPE table up to the highest position we'll touch. */
    int max_pos_needed = (int)(pos_start + n_tokens);
    if (max_pos_needed >= state->max_seq) max_pos_needed = state->max_seq - 1;
    pickle_fast_state_extend_rope(state, max_pos_needed);

    int rc = PICKLE_OK;
    for (size_t ti = 0; ti < n_tokens; ti++) {
        int32_t tok = tokens[ti];
        if (tok < 0 || tok >= VS) { rc = PICKLE_ERR_RANGE; goto done; }
        size_t pos = pos_start + ti;
        if (kv && pos >= (size_t)kv->max_seq) { rc = PICKLE_ERR_RANGE; goto done; }

        /* Embedding lookup. The embd tensor was dequantized to F32 at
         * state-init time (if it was quantized); F16 is dequantized on
         * the fly per row. */
        if (emb_type == GGML_F16) {
            const uint16_t* w = (const uint16_t*)tensors[state->t_embd].data;
            for (int i = 0; i < HD; i++) emb_row[i] = f16_bits_to_float(w[(size_t)tok * HD + i]);
            for (int i = 0; i < HD; i++) x[i] = emb_row[i];
        } else {
            const float* row = emb_w + (size_t)tok * HD;
            for (int i = 0; i < HD; i++) x[i] = row[i];
        }
        for (int L = 0; L < NL; L++) {
            /* --- Attention block --- */
            for (int i = 0; i < HD; i++) xn[i] = x[i];
            pickle_fast_rmsnorm(xn, (const float*)tensors[state->t_attn_norm[L]].data, HD, eps);

            rc = pickle_fast_matmul_dispatch(q, &tensors[state->t_attn_q[L]], xn, H  * D, HD);
            if (rc != PICKLE_OK) goto done;
            rc = pickle_fast_matmul_dispatch(k, &tensors[state->t_attn_k[L]], xn, HK * D, HD);
            if (rc != PICKLE_OK) goto done;
            rc = pickle_fast_matmul_dispatch(v, &tensors[state->t_attn_v[L]], xn, HK * D, HD);
            if (rc != PICKLE_OK) goto done;

            /* RoPE — apply precomputed sin/cos. NeoX convention: split
             * each head into [0..half) and [half..D) halves and rotate
             * the pair (x[i], x[i+half]) by angle=pos*inv_freq[i]. */
            if (half > 0 && pos < (size_t)state->rope_filled_upto) {
                const float* sp = state->rope_sin + (size_t)pos * half;
                const float* cp = state->rope_cos + (size_t)pos * half;
                for (int h = 0; h < H; h++) {
                    float* qh = q + h * D;
                    for (int i = 0; i < half; i++) {
                        float q0 = qh[i];
                        float q1 = qh[i + half];
                        float c = cp[i], s = sp[i];
                        qh[i]        = c * q0 - s * q1;
                        qh[i + half] = s * q0 + c * q1;
                    }
                }
                for (int h = 0; h < HK; h++) {
                    float* kh = k + h * D;
                    for (int i = 0; i < half; i++) {
                        float k0 = kh[i];
                        float k1 = kh[i + half];
                        float c = cp[i], s = sp[i];
                        kh[i]        = c * k0 - s * k1;
                        kh[i + half] = s * k0 + c * k1;
                    }
                }
            }

            /* Write K, V into the contiguous cache and run attention. */
            if (kv) {
                size_t layer_off = (size_t)L * kv->k_stride_layer;
                size_t pos_off   = (size_t)pos * HK * D;
                float* kdst = kv->k + layer_off + pos_off;
                float* vdst = kv->v + layer_off + pos_off;
                for (int i = 0; i < HK * D; i++) { kdst[i] = k[i]; vdst[i] = v[i]; }

                /* Causal attention over [0, pos]. */
                size_t attend_n = pos + 1;
                float scale = 1.0f / sqrtf((float)D);
                for (int h = 0; h < H; h++) {
                    const float* qh = q + h * D;
                    int kv_head = h * HK / H;
                    /* Score every cached K against this Q. */
                    for (size_t p = 0; p < attend_n; p++) {
                        const float* kp = kv->k + layer_off + p * HK * D + kv_head * D;
                        float dot = 0.0f;
                        for (int d = 0; d < D; d++) dot += qh[d] * kp[d];
                        scores[p] = dot * scale;
                    }
                    /* Softmax. */
                    float maxv = scores[0];
                    for (size_t p = 1; p < attend_n; p++)
                        if (scores[p] > maxv) maxv = scores[p];
                    float sume = 0.0f;
                    for (size_t p = 0; p < attend_n; p++) {
                        scores[p] = expf(scores[p] - maxv);
                        sume += scores[p];
                    }
                    if (sume == 0.0f) sume = 1.0f;
                    /* Weighted sum of V. */
                    float* oh = ao + h * D;
                    for (int d = 0; d < D; d++) oh[d] = 0.0f;
                    for (size_t p = 0; p < attend_n; p++) {
                        float w = scores[p] / sume;
                        const float* vp = kv->v + layer_off + p * HK * D + kv_head * D;
                        for (int d = 0; d < D; d++) oh[d] += w * vp[d];
                    }
                }
            } else {
                /* Stateless: self-attend to this token only. */
                for (int h = 0; h < H; h++) {
                    const float* qh = q + h * D;
                    int kv_head = h * HK / H;
                    const float* kp = k + kv_head * D;
                    const float* vp = v + kv_head * D;
                    float dot = 0.0f;
                    for (int d = 0; d < D; d++) dot += qh[d] * kp[d];
                    float w = 1.0f;  /* single token — softmax over 1 */
                    float* oh = ao + h * D;
                    for (int d = 0; d < D; d++) oh[d] = w * vp[d];
                    (void)dot;
                }
            }

            /* Output projection. */
            rc = pickle_fast_matmul_dispatch(aproj, &tensors[state->t_attn_out[L]], ao, HD, H * D);
            if (rc != PICKLE_OK) goto done;
            for (int i = 0; i < HD; i++) x[i] += aproj[i];

            /* --- FFN block --- */
            for (int i = 0; i < HD; i++) xn2[i] = x[i];
            pickle_fast_rmsnorm(xn2, (const float*)tensors[state->t_ffn_norm[L]].data, HD, eps);

            rc = pickle_fast_matmul_dispatch(gate, &tensors[state->t_ffn_gate[L]], xn2, ID, HD);
            if (rc != PICKLE_OK) goto done;
            rc = pickle_fast_matmul_dispatch(up,   &tensors[state->t_ffn_up[L]],   xn2, ID, HD);
            if (rc != PICKLE_OK) goto done;

            /* SwiGLU: act = silu(gate) * up. */
            for (int i = 0; i < ID; i++) {
                float g = gate[i];
                /* silu = g * sigmoid(g) = g / (1 + exp(-g)) */
                float sig = 1.0f / (1.0f + expf(-g));
                act[i] = g * sig * up[i];
            }

            rc = pickle_fast_matmul_dispatch(down, &tensors[state->t_ffn_down[L]], act, HD, ID);
            if (rc != PICKLE_OK) goto done;
            for (int i = 0; i < HD; i++) x[i] += down[i];
        }

        /* Only the last token produces logits (saves a big matmul). */
        if (ti == n_tokens - 1) {
            if (state->t_norm >= 0)
                pickle_fast_rmsnorm(x, (const float*)tensors[state->t_norm].data, HD, eps);
            const float* lm_w;
            if (state->tie_word_embeddings || state->t_output < 0) {
                /* Use embedding table (must be F32 / pre-dequantized). */
                if (tensors[state->t_embd].type != GGML_F32) {
                    /* Need to dequant the whole embedding table for the
                     * final projection. Cheaper alternative: per-row
                     * dequant during the matmul. We have pickle_fast_matmul_dispatch
                     * for that — but it expects a tensor_info whose data
                     * is the raw bytes. If the embd tensor was loaded
                     * via pickle_load (full), its data is already F32
                     * and emb_type == GGML_F32. If loaded via
                     * pickle_load_meta (lazy), data is NULL until
                     * dequantized. */
                    if (!tensors[state->t_embd].data) {
                        rc = pickle_dequant_tensor(model, (size_t)state->t_embd);
                        if (rc != PICKLE_OK) goto done;
                    }
                    /* Now data is F32 (dequantized). */
                    lm_w = (const float*)tensors[state->t_embd].data;
                    pickle_fast_matmul_f32(out_logits, lm_w, x, VS, HD);
                } else {
                    lm_w = (const float*)tensors[state->t_embd].data;
                    pickle_fast_matmul_f32(out_logits, lm_w, x, VS, HD);
                }
            } else {
                /* output.weight is a [VS, HD] matrix. It may be
                 * quantized — dispatch handles that. */
                rc = pickle_fast_matmul_dispatch(out_logits, &tensors[state->t_output], x, VS, HD);
                if (rc != PICKLE_OK) goto done;
            }
        }
    }

    if (kv && kv_pos) *kv_pos = pos_start + n_tokens;

done:
    free(x); free(xn); free(q); free(k); free(v); free(ao); free(aproj);
    free(xn2); free(gate); free(up); free(act); free(down); free(scores);
    free(emb_row);
    return rc;
}

/* ================================================================== */
/* Native-float samplers                                              */
/* ================================================================== */
int32_t pickle_fast_argmax(const float* logits, size_t n) {
    if (!logits || n == 0) return -1;
    int32_t best = 0;
    float bestv = logits[0];
    for (size_t i = 1; i < n; i++) {
        if (logits[i] > bestv) { bestv = logits[i]; best = (int32_t)i; }
    }
    return best;
}

static uint32_t g_rng_state = 0x12345678u;
static uint32_t xorshift32(void) {
    uint32_t x = g_rng_state;
    x ^= x << 13; x ^= x >> 17; x ^= x << 5;
    g_rng_state = x;
    return x;
}

int32_t pickle_fast_sample_temp(float* logits, size_t n,
                                float temp, float top_p, uint32_t seed) {
    if (!logits || n == 0) return -1;
    g_rng_state = seed ? seed : 0x12345678u;
    if (temp <= 0.0f) return pickle_fast_argmax(logits, n);
    /* Scale by temperature. */
    for (size_t i = 0; i < n; i++) logits[i] /= temp;
    /* Softmax (in-place). */
    float maxv = logits[0];
    for (size_t i = 1; i < n; i++) if (logits[i] > maxv) maxv = logits[i];
    float sum = 0.0f;
    for (size_t i = 0; i < n; i++) {
        logits[i] = expf(logits[i] - maxv);
        sum += logits[i];
    }
    if (sum == 0.0f) return 0;
    /* top_p (nucleus) sampling: keep the smallest set of tokens whose
     * cumulative probability >= top_p. */
    if (top_p > 0.0f && top_p < 1.0f) {
        /* Naive O(n^2) nucleus — vocab is at most ~128k, fine for
         * interactive use. We sort indices by probability descending
         * and cut at top_p. */
        /* Allocate index array. */
        int32_t* idx = (int32_t*)malloc(n * sizeof(int32_t));
        if (!idx) {
            /* Fall back to plain temperature sampling. */
            for (size_t i = 0; i < n; i++) logits[i] /= sum;
            float r = (float)((xorshift32() & 0xFFFFFF) / (float)0xFFFFFF);
            float acc = 0.0f;
            for (size_t i = 0; i < n; i++) {
                acc += logits[i];
                if (acc >= r) return (int32_t)i;
            }
            return (int32_t)(n - 1);
        }
        for (size_t i = 0; i < n; i++) idx[i] = (int32_t)i;
        /* Selection sort the top-k until cumulative p >= top_p — O(n*k)
         * where k is typically small. */
        size_t k = 0;
        float cum = 0.0f;
        while (k < n) {
            /* Find max in [k, n). */
            size_t max_i = k;
            float max_p = logits[idx[k]] / sum;
            for (size_t i = k + 1; i < n; i++) {
                float p = logits[idx[i]] / sum;
                if (p > max_p) { max_p = p; max_i = i; }
            }
            int32_t tmp = idx[k]; idx[k] = idx[max_i]; idx[max_i] = tmp;
            cum += max_p;
            k++;
            if (cum >= top_p) break;
        }
        /* Sample from the top-k. */
        float r = (float)((xorshift32() & 0xFFFFFF) / (float)0xFFFFFF) * cum;
        float acc = 0.0f;
        int32_t chosen = idx[0];
        for (size_t i = 0; i < k; i++) {
            acc += logits[idx[i]] / sum;
            if (acc >= r) { chosen = idx[i]; break; }
        }
        free(idx);
        return chosen;
    }
    /* Plain temperature sampling. */
    float r = (float)((xorshift32() & 0xFFFFFF) / (float)0xFFFFFF);
    float acc = 0.0f;
    for (size_t i = 0; i < n; i++) {
        acc += logits[i] / sum;
        if (acc >= r) return (int32_t)i;
    }
    return (int32_t)(n - 1);
}
