/*
 * Lestra OS — Pickle host fast path.
 * Copyright (c) 2026 lestramk.org / Lee Muriihi Kingori
 *
 * This header declares the host-only "fast" API: native C float arithmetic
 * (SSE/AVX auto-vectorised by the compiler), a real Llama BPE tokenizer,
 * quantized-Q4_K dot-product matmul, precomputed RoPE tables, and a
 * contiguous KV cache. The kernel build does not link this file — it
 * keeps using the soft-float path in pickle_softfp.c.
 *
 * The fast path lives next to the soft-float path inside the SAME source
 * tree. pickle_forward() dispatches:
 *
 *     #ifndef PICKLE_KERNEL  → pickle_fast_forward()   (this file)
 *     #else                  → soft-float forward      (pickle.c)
 *
 * Both paths share the same GGUF parser, the same model handle, and the
 * same architecture descriptor, so the soft-float kernel selftest and
 * the fast host inference see identical weights.
 */
#ifndef LESTRA_PICKLE_FAST_H
#define LESTRA_PICKLE_FAST_H

#ifndef PICKLE_KERNEL
#include "pickle.h"
#include <stdio.h>   /* FILE* for pickle_io_init_file declaration */

#ifdef __cplusplus
extern "C" {
#endif

/* Host-only POSIX shim: wire up a pickle_io_t to a FILE*. Used by
 * pickle_fast_state_init to dequantize the embedding from mmap'd raw
 * bytes via fmemopen + pickle_dequant_stream. */
void pickle_io_init_file(pickle_io_t* io, FILE* f);

/* ------------------------------------------------------------------ */
/* Precomputed per-model inference state                              */
/* ------------------------------------------------------------------ */
/* Resolves all per-layer tensor indices once, allocates the RoPE
 * sin/cos table, and the per-head inverse-frequency table. The same
 * state is reused across every forward pass for the lifetime of the
 * model. Call once after pickle_arch_detect(); free with
 * pickle_fast_state_free().
 *
 * Holds:
 *   - resolved tensor indices for embd/norm/output and, per layer,
 *     attn_norm / attn_q / attn_k / attn_v / attn_output / ffn_norm /
 *     ffn_gate / ffn_up / ffn_down
 *   - rope_inv_freq[i] = 1.0 / theta^(2i/D)  for i in [0, head_dim/2)
 *   - rope_sin[pos*half + i], rope_cos[pos*half + i] for pos in
 *     [0, max_seq) — lazily filled up to the highest position seen
 *     (extend on demand with pickle_fast_state_extend_rope()).
 */
typedef struct {
    int  n_layers;
    int  n_heads;
    int  n_kv_heads;
    int  head_dim;
    int  hidden_dim;
    int  intermediate_dim;
    int  vocab_size;
    int  max_seq;
    int  tie_word_embeddings;
    float rope_theta;
    float rms_eps;

    /* resolved tensor indices (>=0) or negative if absent */
    int t_embd;
    int t_norm;
    int t_output;
    int* t_attn_norm;   /* [n_layers] */
    int* t_attn_q;      /* [n_layers] */
    int* t_attn_k;      /* [n_layers] */
    int* t_attn_v;      /* [n_layers] */
    int* t_attn_out;    /* [n_layers] */
    int* t_ffn_norm;    /* [n_layers] */
    int* t_ffn_gate;    /* [n_layers] */
    int* t_ffn_up;      /* [n_layers] */
    int* t_ffn_down;    /* [n_layers] */

    /* Cached tensor info snapshots — filled at state_init time so the
     * forward pass never calls pickle_tensor_info (which copies) and
     * never reaches into the opaque model struct. Each entry is the
     * resolved index into the model's tensor table; the snapshot lives
     * in the tensors[] array below, indexed by the same value. */
    int                  n_tensors;     /* = model->tensor_count */
    pickle_tensor_info_t* tensors;      /* [n_tensors] cached info */

    /* RoPE precompute */
    float* rope_inv_freq;   /* [head_dim/2] */
    float* rope_sin;        /* [max_seq * half] */
    float* rope_cos;        /* [max_seq * half] */
    int    rope_filled_upto; /* highest pos+1 whose sin/cos are filled */
    int    half;             /* head_dim / 2 */

    /* float view of every dequantized tensor (point into model's
     * tensor data buffer which already stores IEEE-754 f32). */

    /* ---- Pre-allocated working buffers for the forward pass ---- */
    /* Allocated once in pickle_fast_state_init() and reused across every
     * forward call. Eliminates ~13 malloc/free pairs per token in the
     * decode loop — the #1 source of allocator overhead. Each buffer is
     * sized for the model's dimensions and never grows. */
    float* x;        /* [hidden_dim]              — current hidden state */
    float* xn;       /* [hidden_dim]              — normed input for q/k/v */
    float* q;        /* [n_heads * head_dim]      — query projection */
    float* k;        /* [n_kv_heads * head_dim]   — key projection */
    float* v;        /* [n_kv_heads * head_dim]   — value projection */
    float* ao;       /* [n_heads * head_dim]      — attention output */
    float* aproj;    /* [hidden_dim]              — attention output projection */
    float* xn2;      /* [hidden_dim]              — normed input for gate/up */
    float* gate;     /* [intermediate_dim]        — FFN gate projection */
    float* up;       /* [intermediate_dim]        — FFN up projection */
    float* act;      /* [intermediate_dim]        — SwiGLU activation */
    float* down;     /* [hidden_dim]              — FFN down projection */
    float* scores;   /* [max_seq]                 — attention score scratch */
    float* emb_row;  /* [hidden_dim]              — per-row F16 embd dequant */
    float* dequant_embd; /* host: F32 copy of token_embd when it was
                           * quantized and loaded via mmap (mmap patches
                           * t->data to raw quant bytes, but the forward
                           * pass reads embd as F32). NULL if not needed. */
} pickle_fast_state_t;

/* Build the fast-state from a loaded model + detected arch. Returns
 * PICKLE_OK or a negative error. On error, *state is zeroed. */
int  pickle_fast_state_init(const pickle_model_t* m,
                            const pickle_arch_t* arch,
                            pickle_fast_state_t* state);
void pickle_fast_state_free(pickle_fast_state_t* state);

/* Ensure rope_sin/rope_cos are filled up to (and including) position
 * `upto_pos`. Returns PICKLE_OK or PICKLE_ERR_MEMORY. */
int  pickle_fast_state_extend_rope(pickle_fast_state_t* state, int upto_pos);


/* ------------------------------------------------------------------ */
/* Contiguous KV cache                                                */
/* ------------------------------------------------------------------ */
/* A single-allocation KV cache: K and V are each one big
 * [n_layers * max_seq * n_kv_heads * head_dim] float buffer.
 * Far better cache locality than pickle_kv_cache_t's per-slot mallocs
 * and avoids hundreds of thousands of allocator calls at startup.
 */
typedef struct {
    int    n_layers;
    int    n_kv_heads;
    int    head_dim;
    int    max_seq;
    size_t k_stride_layer;   /* max_seq * n_kv_heads * head_dim */
    size_t v_stride_layer;
    float* k;                /* [n_layers * k_stride_layer] */
    float* v;                /* [n_layers * v_stride_layer] */
} pickle_fast_kv_t;

int  pickle_fast_kv_alloc(const pickle_arch_t* arch, pickle_fast_kv_t* kv);
void pickle_fast_kv_free (pickle_fast_kv_t* kv);


/* ------------------------------------------------------------------ */
/* Fast forward pass                                                  */
/* ------------------------------------------------------------------ */
/* Run n_tokens tokens through the Llama forward pass using native
 * C float arithmetic (auto-vectorised to SSE/AVX by the compiler).
 *
 * Tokens are processed one at a time (decode path is what matters for
 * chat — prefill batching is a v0.5 roadmap item). The KV cache is
 * written starting at *kv_pos (in: where to start; out: new position).
 * Logits for the LAST token only are written to out_logits (must be
 * arch->vocab_size floats).
 *
 * state must have been initialised with pickle_fast_state_init().
 * kv may be NULL for stateless single-token inference (then kv_pos
 * is ignored).
 */
int pickle_fast_forward(pickle_model_t*        model,
                        const pickle_arch_t*   arch,
                        pickle_fast_state_t*   state,
                        const int32_t*         tokens,
                        size_t                 n_tokens,
                        float*                 out_logits,
                        pickle_fast_kv_t*      kv,
                        size_t*                kv_pos);


/* ------------------------------------------------------------------ */
/* Native-float math helpers (also used by the tokenizer / sampler)   */
/* ------------------------------------------------------------------ */
void pickle_fast_rmsnorm(float* x, const float* w, int n, float eps);
void pickle_fast_matmul_f32(float* y, const float* W, const float* x,
                            int out_n, int in_n);

/* Quantized Q4_K matmul: W is the raw on-disk Q4_K block buffer
 * (NOT pre-dequantized), x is F32 input of length in_n. Writes out_n
 * F32 outputs to y. Block layout matches GGML Q4_K (144 bytes / 256
 * elements). in_n and out_n must be multiples of 256. */
void pickle_fast_matmul_q4_k(float* y,
                             const unsigned char* W_blocks,
                             int out_n, int in_n,
                             const float* x);

/* Quantized Q8_0 matmul — same shape, 34 bytes / 32 elements. */
void pickle_fast_matmul_q8_0(float* y,
                             const unsigned char* W_blocks,
                             int out_n, int in_n,
                             const float* x);

/* Quantized Q4_0 matmul — 18 bytes / 32 elements. */
void pickle_fast_matmul_q4_0(float* y,
                             const unsigned char* W_blocks,
                             int out_n, int in_n,
                             const float* x);

/* Quantized Q6_K matmul — 210 bytes / 256 elements. */
void pickle_fast_matmul_q6_k(float* y,
                             const unsigned char* W_blocks,
                             int out_n, int in_n,
                             const float* x);

/* Quantized Q5_K matmul — 176 bytes / 256 elements. */
void pickle_fast_matmul_q5_k(float* y,
                             const unsigned char* W_blocks,
                             int out_n, int in_n,
                             const float* x);

/* Quantized Q5_0 matmul — 22 bytes / 32 elements. */
void pickle_fast_matmul_q5_0(float* y,
                             const unsigned char* W_blocks,
                             int out_n, int in_n,
                             const float* x);

/* Quantized Q5_1 matmul — 24 bytes / 32 elements. */
void pickle_fast_matmul_q5_1(float* y,
                             const unsigned char* W_blocks,
                             int out_n, int in_n,
                             const float* x);

/* Quantized Q4_1 matmul — 20 bytes / 32 elements. */
void pickle_fast_matmul_q4_1(float* y,
                             const unsigned char* W_blocks,
                             int out_n, int in_n,
                             const float* x);

/* Quantized Q8_1 matmul — 36 bytes / 32 elements. */
void pickle_fast_matmul_q8_1(float* y,
                             const unsigned char* W_blocks,
                             int out_n, int in_n,
                             const float* x);

/* Quantized Q8_K matmul — 292 bytes / 256 elements. */
void pickle_fast_matmul_q8_k(float* y,
                             const unsigned char* W_blocks,
                             int out_n, int in_n,
                             const float* x);

/* Quantized Q2_K matmul — 84 bytes / 256 elements. */
void pickle_fast_matmul_q2_k(float* y,
                             const unsigned char* W_blocks,
                             int out_n, int in_n,
                             const float* x);

/* Quantized Q3_K matmul — 110 bytes / 256 elements. */
void pickle_fast_matmul_q3_k(float* y,
                             const unsigned char* W_blocks,
                             int out_n, int in_n,
                             const float* x);

/* Quantized F16 matmul — 2 bytes / element. */
void pickle_fast_matmul_f16(float* y,
                            const unsigned char* W_bytes,
                            int out_n, int in_n,
                            const float* x);

/* Dispatch matmul by tensor type. W_blocks is the raw on-disk bytes
 * for the weight tensor (NOT dequantized). Returns PICKLE_OK or
 * PICKLE_ERR_TYPE if the type has no fast matmul. */
int pickle_fast_matmul_dispatch(float* y,
                                const pickle_tensor_info_t* t,
                                const float* x,
                                int out_n, int in_n);


/* ------------------------------------------------------------------ */
/* Llama BPE tokenizer                                                */
/* ------------------------------------------------------------------ */
/* A real SentencePiece-style BPE tokenizer that reads
 * tokenizer.ggml.tokens / .scores / .merges / .model from the GGUF
 * metadata. Supports Llama-2 / Llama-3 / Mistral / Qwen2 / TinyLlama
 * style vocabularies with byte-fallback.
 *
 * Encode: text → token ids (with BOS prepended if add_bos).
 * Decode: token ids → text (special tokens emitted as their literal
 *         text, byte-fallback tokens decoded back to UTF-8 bytes).
 */
typedef struct pickle_tokenizer pickle_tokenizer_t;

/* Build a tokenizer from a loaded model's metadata. Returns PICKLE_OK
 * or a negative error. On success *out_tok is a new handle (free with
 * pickle_tok_free). Returns PICKLE_ERR_ARCH if no vocab metadata is
 * present. */
int  pickle_tok_init(const pickle_model_t* m, pickle_tokenizer_t** out_tok);
void pickle_tok_free(pickle_tokenizer_t* tok);

/* Encode text into token ids. *out_ids is set to a malloc'd array
 * (caller frees), *out_n is the count. If add_bos != 0, prepend the
 * BOS token id (tokenizer.ggml.bos_token_id or 1). */
int  pickle_tok_encode(pickle_tokenizer_t* tok,
                       const char* text, int add_bos,
                       int32_t** out_ids, size_t* out_n);

/* Decode token ids into text. *out_text is a malloc'd NUL-terminated
 * string (caller frees). skip_special != 0 → special tokens dropped. */
int  pickle_tok_decode(pickle_tokenizer_t* tok,
                       const int32_t* ids, size_t n,
                       int skip_special,
                       char** out_text);

/* Look up token id by literal token text (e.g. "<eos>"). Returns -1
 * if not found. */
int  pickle_tok_id(const pickle_tokenizer_t* tok, const char* token_text);

/* Return the token text for an id (may be a special token or a
 * byte-fallback byte). Returns NULL if id is out of range. The
 * returned pointer is owned by the tokenizer. */
const char* pickle_tok_text(const pickle_tokenizer_t* tok, int32_t id);

/* Vocab size (== arch.vocab_size for well-formed models). */
int  pickle_tok_size(const pickle_tokenizer_t* tok);

/* BOS / EOS / UNK / pad token ids (-1 if unknown). */
int  pickle_tok_bos(const pickle_tokenizer_t* tok);
int  pickle_tok_eos(const pickle_tokenizer_t* tok);
int  pickle_tok_unk(const pickle_tokenizer_t* tok);
int  pickle_tok_pad(const pickle_tokenizer_t* tok);


/* ------------------------------------------------------------------ */
/* Native-float samplers                                              */
/* ------------------------------------------------------------------ */
int32_t pickle_fast_argmax     (const float* logits, size_t n);
int32_t pickle_fast_sample_temp(float* logits, size_t n,
                                float temp, float top_p, uint32_t seed);


#ifdef __cplusplus
}
#endif
#endif /* !PICKLE_KERNEL */
#endif /* LESTRA_PICKLE_FAST_H */
