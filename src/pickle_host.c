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
 *   - pickle_run_prompt(model, arch, prompt_str, n_gen, out_buf, buf_size)
 *     — runs the Llama forward pass on a character-level-tokenized prompt,
 *     then greedily generates n_gen tokens and returns them in out_buf
 *
 * This file is host-only: it uses <stdio.h>, <stdlib.h>, <string.h> and is
 * NOT compiled for the lestraOS kernel.
 */
#include "pickle.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>

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
