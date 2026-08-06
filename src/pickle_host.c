/*
 * pickle_host.c — POSIX shim for the pickle GGUF engine.
 * Copyright (c) 2026 lestramk.org / Lee Muriihi Kingori
 *
 * Bridges the freestanding pickle core (pickle.c / pickle_softfp.c /
 * pickle.h — the SAME source that compiles into the lestraOS kernel with
 * -DPICKLE_KERNEL) to a normal POSIX host environment.
 *
 * Provides:
 *   - pickle_io_t callbacks wrapping a host FILE*  (fread / fseek / ftell)
 *   - pickle_alloc_t wrapping malloc / free
 *   - pickle_load_from_file(path, &model)  — opens the file, wires up the
 *     io + alloc, and calls pickle_load
 *   - pickle_load_from_file_lazy(path, &model) — metadata-only load via
 *     pickle_load_meta(); the FILE* is kept open for on-demand dequant.
 *     Used by `chat` / `bench` so we don't pay the full-dequant cost up
 *     front — fast matmul dequants each block on the fly inside the
 *     dot product.
 *   - pickle_run_prompt(model, arch, prompt_str, n_gen, out_buf, buf_size)
 *     — runs the Llama forward pass on a character-level-tokenized prompt,
 *     then greedily generates n_gen tokens and returns them in out_buf
 *     (legacy soft-float path; kept for the demo model)
 *   - pickle_chat_loop(...) — fast-path streaming generation loop with
 *     BPE tokenizer, contiguous KV cache, optional temperature/top_p
 *     sampling, and a per-token callback for streaming output.
 *
 * This file is host-only: it uses <stdio.h>, <stdlib.h>, <string.h> and is
 * NOT compiled for the lestraOS kernel.
 */
#include "pickle.h"
#include "pickle_fast.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <errno.h>
#include <time.h>

/* ================================================================== */
/* FILE*-based pickle_io_t                                            */
/* ================================================================== */
static size_t host_io_read(void* ctx, void* buf, size_t len) {
    FILE* f = (FILE*)ctx;
    if (!f || !buf || len == 0) return 0;
    return fread(buf, 1, len, f);
}
static int host_io_seek(void* ctx, int64_t offset, int whence) {
    FILE* f = (FILE*)ctx;
    if (!f) return -1;
    int w;
    switch (whence) {
        case PICKLE_SEEK_SET: w = SEEK_SET; break;
        case PICKLE_SEEK_CUR: w = SEEK_CUR; break;
        case PICKLE_SEEK_END: w = SEEK_END; break;
        default: return -1;
    }
    /* fseek takes a long; clamp to a sane range */
    if (offset > 0x7fffffffLL) offset = 0x7fffffffLL;
    if (offset < -0x7fffffffLL) offset = -0x7fffffffLL;
    return fseek(f, (long)offset, w);
}
static int64_t host_io_tell(void* ctx) {
    FILE* f = (FILE*)ctx;
    if (!f) return -1;
    long p = ftell(f);
    return (int64_t)p;
}

static void pickle_io_init_file(pickle_io_t* io, FILE* f) {
    io->ctx   = (void*)f;
    io->read  = host_io_read;
    io->seek  = host_io_seek;
    io->tell  = host_io_tell;
}

/* ================================================================== */
/* malloc-based pickle_alloc_t                                        */
/* ================================================================== */
static void* host_alloc(void* ctx, size_t size) {
    (void)ctx;
    /* pickle_set_alloc expects zeroed memory (mirrors the kernel bump
     * allocator's behaviour). calloc() gives us that for free. */
    if (size == 0) size = 1;
    return calloc(size, 1);
}
static void host_free(void* ctx, void* ptr, size_t size) {
    (void)ctx;
    (void)size;
    free(ptr);
}

static void pickle_alloc_init_host(void) {
    pickle_alloc_t a;
    a.ctx   = 0;
    a.alloc = host_alloc;
    a.free  = host_free;
    pickle_set_alloc(&a);
}

/* ================================================================== */
/* pickle_load_from_file                                              */
/* ================================================================== */
int pickle_load_from_file(const char* path, pickle_model_t** out_model) {
    if (!path || !out_model) return PICKLE_ERR_ARG;
    *out_model = 0;

    pickle_alloc_init_host();

    FILE* f = fopen(path, "rb");
    if (!f) {
        fprintf(stderr, "pickle: failed to open '%s'\n", path);
        return PICKLE_ERR_IO;
    }

    pickle_io_t io;
    pickle_io_init_file(&io, f);

    int rc = pickle_load(&io, out_model);
    fclose(f);
    if (rc != PICKLE_OK) {
        fprintf(stderr, "pickle: failed to load model '%s': rc=%d\n", path, rc);
        return rc;
    }
    return PICKLE_OK;
}

/* Metadata-only load: parses header + KV + tensor table but does NOT
 * dequantize tensor data. The FILE* is kept open (stored in the model's
 * io) so individual tensors can be dequantized on demand via
 * pickle_dequant_tensor(). The caller must still call pickle_free()
 * (which will close the file via the io — see pickle_host_close).
 *
 * Use this for `info` and `dequant <tensor>` — instant vs hours. */
static void host_io_close(void* ctx) {
    FILE* f = (FILE*)ctx;
    if (f) fclose(f);
}

int pickle_load_from_file_meta(const char* path, pickle_model_t** out_model) {
    if (!path || !out_model) return PICKLE_ERR_ARG;
    *out_model = 0;

    pickle_alloc_init_host();

    FILE* f = fopen(path, "rb");
    if (!f) {
        fprintf(stderr, "pickle: failed to open '%s'\n", path);
        return PICKLE_ERR_IO;
    }

    /* Allocate a persistent io struct (not stack) so it outlives this
     * function call. pickle_free() will close the file and free the io. */
    pickle_io_t* io = (pickle_io_t*)calloc(1, sizeof(pickle_io_t));
    if (!io) { fclose(f); return PICKLE_ERR_MEMORY; }
    pickle_io_init_file(io, f);
    io->close = host_io_close;  /* pickle_free will call this */

    int rc = pickle_load_meta(io, out_model);
    if (rc != PICKLE_OK) {
        fprintf(stderr, "pickle: failed to load meta '%s': rc=%d\n", path, rc);
        fclose(f);
        free(io);
        return rc;
    }
    return PICKLE_OK;
}

/* ================================================================== */
/* pickle_run_prompt                                                  */
/* ================================================================== */
/* Character-level tokenizer for the demo model: each prompt byte becomes
 * its own token id (mod vocab_size). This is sufficient for the demo
 * model (16-vocab F32 Llama), whose forward pass we want to exercise
 * end-to-end on the host. A real BPE tokenizer is a v0.4 roadmap item.
 *
 * Generates n_gen tokens greedily (argmax of logits) and appends each to
 * the KV cache. Writes the generated token ids into out_buf (caller-owned,
 * must have room for *out_n tokens). Returns PICKLE_OK on success.
 */
int pickle_run_prompt(
    pickle_model_t*        model,
    const pickle_arch_t*   arch,
    const char*            prompt_str,
    size_t                 n_gen,
    int32_t*               out_buf,
    size_t                 out_buf_size,
    size_t*                out_n
) {
    if (!model || !arch || !prompt_str || !out_buf) return PICKLE_ERR_ARG;
    if (n_gen == 0) { if (out_n) *out_n = 0; return PICKLE_OK; }
    if (out_buf_size < n_gen) return PICKLE_ERR_RANGE;

    int VS = arch->vocab_size;
    if (VS <= 0) return PICKLE_ERR_ARCH;

    /* Tokenize the prompt: ASCII byte values, clamped to [0, VS). */
    size_t prompt_len = strlen(prompt_str);
    if (prompt_len == 0) {
        /* Use a single fixed token if the prompt is empty so we still
         * produce a logits vector. */
        prompt_len = 1;
    }
    int32_t* prompt_tokens = (int32_t*)calloc(prompt_len, sizeof(int32_t));
    if (!prompt_tokens) return PICKLE_ERR_MEMORY;
    if (strlen(prompt_str) == 0) {
        prompt_tokens[0] = 0;
    } else {
        for (size_t i = 0; i < prompt_len; i++) {
            int b = ((const unsigned char*)prompt_str)[i];
            prompt_tokens[i] = (int32_t)(b % VS);
        }
    }

    /* Allocate KV cache big enough for prompt + n_gen. */
    pickle_kv_cache_t kv;
    int rc = pickle_kv_alloc(arch, &kv);
    if (rc != PICKLE_OK) { free(prompt_tokens); return rc; }

    /* logits buffer: one vocab vector. */
    float* logits = (float*)calloc((size_t)VS, sizeof(float));
    if (!logits) { pickle_kv_free(&kv); free(prompt_tokens); return PICKLE_ERR_MEMORY; }

    /* Forward pass over the full prompt — writes logits for the LAST
     * prompt token, and populates the KV cache. */
    size_t kv_pos = 0;
    rc = pickle_forward(model, arch, prompt_tokens, prompt_len, logits, &kv, &kv_pos);
    if (rc != PICKLE_OK) {
        free(logits); pickle_kv_free(&kv); free(prompt_tokens);
        return rc;
    }

    /* Greedy generation loop. Each iteration:
     *   - sample one token from logits (argmax)
     *   - feed it back into the KV cache as a single-token forward pass
     *     (kv_pos already advanced past the prompt), producing logits
     *     for the NEXT position
     *   - stop early if we exceed max_seq_len
     */
    size_t generated = 0;
    for (size_t i = 0; i < n_gen; i++) {
        int32_t tok = pickle_argmax(logits, (size_t)VS);
        if (tok < 0) tok = 0;
        out_buf[generated++] = tok;

        if (kv_pos >= (size_t)arch->max_seq_len) {
            /* KV cache full — stop generating. */
            break;
        }

        int32_t one = tok;
        rc = pickle_forward(model, arch, &one, 1, logits, &kv, &kv_pos);
        if (rc != PICKLE_OK) {
            free(logits); pickle_kv_free(&kv); free(prompt_tokens);
            return rc;
        }
    }

    if (out_n) *out_n = generated;
    free(logits);
    pickle_kv_free(&kv);
    free(prompt_tokens);
    return PICKLE_OK;
}

/* ================================================================== */
/* Lazy (metadata-only) load — for chat / bench                       */
/* ================================================================== */
/* Same as pickle_load_from_file_meta() but kept open as the model's
 * io so on-demand dequant works. Used by chat/bench/tokens so we
 * don't materialise the full F32 weight matrix. */
int pickle_load_from_file_lazy(const char* path, pickle_model_t** out_model) {
    return pickle_load_from_file_meta(path, out_model);
}

/* ================================================================== */
/* Fast chat loop — streaming generation with BPE tokenizer           */
/* ================================================================== */
/* Run the fast-path forward pass over a prompt (BPE-tokenised by the
 * host tokenizer), then generate up to max_new_tokens, invoking the
 * callback after each token. The callback may return non-zero to
 * stop generation (e.g. on EOS or user interrupt).
 *
 * If temp <= 0, uses greedy argmax. Otherwise uses temperature +
 * top_p (nucleus) sampling with the given seed.
 *
 * Stats: *out_prompt_n and *out_generated_n are filled with the
 * prompt token count and the number of generated tokens (may be
 * NULL). Returns PICKLE_OK on success.
 */
int pickle_chat_loop(
    pickle_model_t*        model,
    const pickle_arch_t*   arch,
    pickle_tokenizer_t*    tok,
    const char*            prompt,
    int                    add_bos,
    size_t                 max_new_tokens,
    float                  temp,
    float                  top_p,
    uint32_t               seed,
    /* callback: invoked with each generated token id; return non-zero
     * to stop generation. user_ctx is passed through. */
    int                    (*on_token)(int32_t id, void* user_ctx),
    void*                  user_ctx,
    size_t*                out_prompt_n,
    size_t*                out_generated_n,
    double*                out_prefill_ms,
    double*                out_decode_avg_ms
) {
    if (!model || !arch || !tok || !prompt) return PICKLE_ERR_ARG;

    int rc;
    pickle_fast_state_t state;
    rc = pickle_fast_state_init(model, arch, &state);
    if (rc != PICKLE_OK) return rc;

    pickle_fast_kv_t kv;
    rc = pickle_fast_kv_alloc(arch, &kv);
    if (rc != PICKLE_OK) { pickle_fast_state_free(&state); return rc; }

    /* Tokenise prompt. */
    int32_t* prompt_ids = NULL;
    size_t   prompt_n = 0;
    rc = pickle_tok_encode(tok, prompt, add_bos, &prompt_ids, &prompt_n);
    if (rc != PICKLE_OK) {
        pickle_fast_kv_free(&kv);
        pickle_fast_state_free(&state);
        return rc;
    }
    if (prompt_n == 0) {
        free(prompt_ids);
        pickle_fast_kv_free(&kv);
        pickle_fast_state_free(&state);
        return PICKLE_ERR_ARG;
    }

    /* The embedding tensor is dequantized inside pickle_fast_state_init
     * (which has already been called above) if it was quantized. So by
     * here, the cached state->tensors[embd].data is a valid F32 pointer. */

    int VS = arch->vocab_size;
    float* logits = (float*)calloc((size_t)VS, sizeof(float));
    if (!logits) {
        free(prompt_ids);
        pickle_fast_kv_free(&kv);
        pickle_fast_state_free(&state);
        return PICKLE_ERR_MEMORY;
    }

    /* ---- Prefill: run the whole prompt through in one call. ---- */
    struct timespec t0, t1;
    clock_gettime(CLOCK_MONOTONIC, &t0);
    size_t kv_pos = 0;
    rc = pickle_fast_forward(model, arch, &state,
                             prompt_ids, prompt_n,
                             logits, &kv, &kv_pos);
    clock_gettime(CLOCK_MONOTONIC, &t1);
    double prefill_ms = (t1.tv_sec - t0.tv_sec) * 1000.0 +
                        (t1.tv_nsec - t0.tv_nsec) / 1e6;
    if (rc != PICKLE_OK) {
        free(logits); free(prompt_ids);
        pickle_fast_kv_free(&kv);
        pickle_fast_state_free(&state);
        return rc;
    }

    /* ---- Decode loop ---- */
    size_t generated = 0;
    double decode_total_ms = 0.0;
    int eos_id = pickle_tok_eos(tok);
    int stop = 0;
    for (size_t i = 0; i < max_new_tokens && !stop; i++) {
        int32_t next;
        if (temp <= 0.0f) {
            next = pickle_fast_argmax(logits, (size_t)VS);
        } else {
            /* The sampler mutates logits (applies softmax in place).
             * That's fine — we recompute logits on the next forward
             * pass anyway. */
            next = pickle_fast_sample_temp(logits, (size_t)VS, temp, top_p, seed + (uint32_t)i);
        }
        if (next < 0) next = 0;

        if (on_token) {
            if (on_token(next, user_ctx) != 0) { stop = 1; }
        }
        generated++;
        if (next == eos_id) { stop = 1; }
        if (stop) break;

        if (kv_pos >= (size_t)arch->max_seq_len) break;

        int32_t one = next;
        clock_gettime(CLOCK_MONOTONIC, &t0);
        rc = pickle_fast_forward(model, arch, &state,
                                 &one, 1,
                                 logits, &kv, &kv_pos);
        clock_gettime(CLOCK_MONOTONIC, &t1);
        decode_total_ms += (t1.tv_sec - t0.tv_sec) * 1000.0 +
                           (t1.tv_nsec - t0.tv_nsec) / 1e6;
        if (rc != PICKLE_OK) break;
    }

    free(logits);
    free(prompt_ids);
    pickle_fast_kv_free(&kv);
    pickle_fast_state_free(&state);

    if (out_prompt_n)      *out_prompt_n      = prompt_n;
    if (out_generated_n)   *out_generated_n   = generated;
    if (out_prefill_ms)    *out_prefill_ms    = prefill_ms;
    if (out_decode_avg_ms) *out_decode_avg_ms = generated > 1
        ? decode_total_ms / (double)(generated - 1)
        : 0.0;
    return rc;
}
