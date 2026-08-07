/*
 * Lestra OS — Pickle Llama BPE tokenizer.
 * Copyright (c) 2026 lestramk.org / Lee Muriithi Kingori
 *
 * Reads the standard Llama / SentencePiece BPE vocab from GGUF
 * metadata and implements encode + decode with byte-fallback.
 *
 * Metadata keys used (all standard, present in Llama / Mistral / Qwen2
 * / TinyLlama GGUFs):
 *
 *   tokenizer.ggml.model         = "llama" | "gpt2"   (we treat both
 *                                                       as BPE)
 *   tokenizer.ggml.tokens        = [string] * vocab_size
 *   tokenizer.ggml.scores        = [float32] * vocab_size
 *   tokenizer.ggml.merges        = [string] * (vocab_size - 256)   [optional]
 *   tokenizer.ggml.token_type    = [int32]   * vocab_size          [optional]
 *   tokenizer.ggml.bos_token_id  = uint32
 *   tokenizer.ggml.eos_token_id  = uint32
 *   tokenizer.ggml.unknown_token_id = uint32
 *   tokenizer.ggml.padding_token_id = uint32
 *   tokenizer.ggml.add_bos_token    = bool
 *   tokenizer.ggml.add_eos_token    = bool
 *
 * The tokenizer handles:
 *   - The SentencePiece "▁" (U+2581) → space convention.
 *   - Byte-fallback tokens "<0x00>".."<0xFF>" — when a piece can't be
 *     merged, the encoder falls back to emitting raw byte tokens.
 *   - Special tokens (token_type == 3) — emitted literally and never
 *     merged.
 *
 * Encode is the classic "split on whitespace, byte-fallback per word,
 * greedily merge highest-scoring pairs" SentencePiece algorithm.
 */
#include "pickle_fast.h"

#include <stdint.h>
#include <stddef.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

/* ------------------------------------------------------------------ */
/* Token type flags (from ggml)                                       */
/* ------------------------------------------------------------------ */
#define TOK_TYPE_NORMAL       1
#define TOK_TYPE_UNKNOWN      2
#define TOK_TYPE_CONTROL      3
#define TOK_TYPE_USER_DEFINED 4
#define TOK_TYPE_UNUSED       5
#define TOK_TYPE_BYTE         6

/* ------------------------------------------------------------------ */
/* Tokenizer struct                                                   */
/* ------------------------------------------------------------------ */
struct pickle_tokenizer {
    int       vocab_size;
    char**    tokens;       /* [vocab_size] NUL-terminated piece strings */
    float*    scores;       /* [vocab_size] merge priority (higher = first) */
    uint8_t*  types;        /* [vocab_size] token_type */
    int*      byte_to_id;   /* 256 entry map; byte_id = -1 if no <0xXX> token */

    /* Hash map: token string -> id. Linear-probing, FNV-1a hash. */
    uint32_t* hash_keys;    /* [hash_cap] hashes (0 = empty) */
    int32_t*  hash_vals;    /* [hash_cap] token id (-1 = empty) */
    size_t    hash_cap;
    size_t    hash_count;

    int bos_id;
    int eos_id;
    int unk_id;
    int pad_id;
    int add_bos;
    int add_eos;
};

/* ------------------------------------------------------------------ */
/* FNV-1a hash                                                        */
/* ------------------------------------------------------------------ */
static uint32_t fnv1a(const char* s, size_t n) {
    uint32_t h = 0x811c9dc5u;
    for (size_t i = 0; i < n; i++) {
        h ^= (uint32_t)(unsigned char)s[i];
        h *= 0x01000193u;
    }
    return h ? h : 1u;   /* never 0 — 0 means empty slot */
}

static int tok_hash_insert(pickle_tokenizer_t* t, const char* s, size_t n, int id) {
    if (t->hash_count * 2 >= t->hash_cap) {
        /* Grow. */
        size_t old_cap = t->hash_cap;
        uint32_t* old_keys = t->hash_keys;
        int32_t*  old_vals = t->hash_vals;
        t->hash_cap = old_cap ? old_cap * 2 : 65536;
        t->hash_keys = (uint32_t*)calloc(t->hash_cap, sizeof(uint32_t));
        t->hash_vals = (int32_t*)malloc(t->hash_cap * sizeof(int32_t));
        if (!t->hash_keys || !t->hash_vals) { free(old_keys); free(old_vals); return -1; }
        for (size_t i = 0; i < t->hash_cap; i++) t->hash_vals[i] = -1;
        t->hash_count = 0;
        for (size_t i = 0; i < old_cap; i++) {
            if (old_keys[i] != 0) {
                /* Re-insert. */
                uint32_t h = old_keys[i];
                size_t idx = h & (t->hash_cap - 1);
                while (t->hash_keys[idx] != 0) idx = (idx + 1) & (t->hash_cap - 1);
                t->hash_keys[idx] = h;
                t->hash_vals[idx] = old_vals[i];
                t->hash_count++;
            }
        }
        free(old_keys); free(old_vals);
    }
    uint32_t h = fnv1a(s, n);
    size_t idx = h & (t->hash_cap - 1);
    while (t->hash_keys[idx] != 0 && t->hash_vals[idx] != id) {
        if (t->hash_keys[idx] == h) {
            /* Already present — overwrite (shouldn't happen for unique tokens). */
            t->hash_vals[idx] = id;
            return 0;
        }
        idx = (idx + 1) & (t->hash_cap - 1);
    }
    t->hash_keys[idx] = h;
    t->hash_vals[idx] = id;
    t->hash_count++;
    return 0;
}

static int tok_hash_find(const pickle_tokenizer_t* t, const char* s, size_t n) {
    if (t->hash_cap == 0) return -1;
    uint32_t h = fnv1a(s, n);
    size_t idx = h & (t->hash_cap - 1);
    while (t->hash_keys[idx] != 0) {
        if (t->hash_keys[idx] == h) {
            int id = t->hash_vals[idx];
            /* Verify length + content match (handles hash collisions). */
            const char* tok = t->tokens[id];
            if (strlen(tok) == n && memcmp(tok, s, n) == 0) return id;
        }
        idx = (idx + 1) & (t->hash_cap - 1);
    }
    return -1;
}

/* ------------------------------------------------------------------ */
/* Init / free                                                        */
/* ------------------------------------------------------------------ */

/* We need a way to read the GGUF_ARRAY metadata. The public API only
 * exposes scalar lookups. Add a small helper in pickle.c that returns
 * a pointer to the raw array data. We declare it here and define it
 * in pickle.c. */
typedef struct {
    uint32_t type;     /* GGUF value type of element */
    uint64_t n;        /* element count */
    const void* data;  /* raw bytes */
} pickle_array_view_t;

/* Defined in pickle.c — exposes a read-only view of an array KV. */
int pickle_meta_array(const pickle_model_t* m, const char* key, pickle_array_view_t* out);

int pickle_tok_init(const pickle_model_t* m, pickle_tokenizer_t** out_tok) {
    if (!m || !out_tok) return PICKLE_ERR_ARG;
    *out_tok = 0;

    pickle_array_view_t tokens_arr = {0}, scores_arr = {0}, types_arr = {0};
    if (pickle_meta_array(m, "tokenizer.ggml.tokens", &tokens_arr) != PICKLE_OK)
        return PICKLE_ERR_ARCH;
    if (tokens_arr.n == 0) return PICKLE_ERR_ARCH;

    pickle_tokenizer_t* t = (pickle_tokenizer_t*)calloc(1, sizeof(*t));
    if (!t) return PICKLE_ERR_MEMORY;
    t->vocab_size = (int)tokens_arr.n;
    t->tokens = (char**)calloc((size_t)t->vocab_size, sizeof(char*));
    t->scores = (float*)calloc((size_t)t->vocab_size, sizeof(float));
    t->types  = (uint8_t*)calloc((size_t)t->vocab_size, sizeof(uint8_t));
    t->byte_to_id = (int*)malloc(256 * sizeof(int));
    if (!t->tokens || !t->scores || !t->types || !t->byte_to_id) {
        pickle_tok_free(t); return PICKLE_ERR_MEMORY;
    }
    for (int i = 0; i < 256; i++) t->byte_to_id[i] = -1;

    /* tokens[] is an array of GGUF_STRING. The parser stores it as a
     * char** (array of n NUL-terminated string pointers). out->n is
     * the element count, out->data is the char**. */
    if (tokens_arr.type != GGUF_STRING) {
        pickle_tok_free(t);
        return PICKLE_ERR_FORMAT;
    }
    {
        char** arr = (char**)tokens_arr.data;
        for (int i = 0; i < t->vocab_size; i++) {
            const char* s = arr[i];
            size_t slen = s ? strlen(s) : 0;
            char* copy = (char*)malloc(slen + 1);
            if (!copy) { pickle_tok_free(t); return PICKLE_ERR_MEMORY; }
            memcpy(copy, s, slen);
            copy[slen] = 0;
            t->tokens[i] = copy;
        }
    }

    /* scores[] — array of FLOAT32. */
    if (pickle_meta_array(m, "tokenizer.ggml.scores", &scores_arr) == PICKLE_OK &&
        scores_arr.type == GGUF_FLOAT32 && scores_arr.n >= (uint64_t)t->vocab_size) {
        const float* s = (const float*)scores_arr.data;
        for (int i = 0; i < t->vocab_size; i++) t->scores[i] = s[i];
    }

    /* token_type[] — array of INT32 (optional). */
    if (pickle_meta_array(m, "tokenizer.ggml.token_type", &types_arr) == PICKLE_OK &&
        types_arr.type == GGUF_INT32 && types_arr.n >= (uint64_t)t->vocab_size) {
        const int32_t* ty = (const int32_t*)types_arr.data;
        for (int i = 0; i < t->vocab_size; i++) t->types[i] = (uint8_t)ty[i];
    } else {
        for (int i = 0; i < t->vocab_size; i++) t->types[i] = TOK_TYPE_NORMAL;
    }

    /* Build hash map + byte_to_id map. */
    for (int i = 0; i < t->vocab_size; i++) {
        const char* s = t->tokens[i];
        size_t slen = strlen(s);
        tok_hash_insert(t, s, slen, i);
        /* Recognise "<0xXX>" byte-fallback tokens. */
        if (slen == 6 && s[0] == '<' && s[1] == '0' && s[2] == 'x' &&
            s[5] == '>') {
            int hi = (s[3] >= '0' && s[3] <= '9') ? s[3] - '0' : (s[3] >= 'a' && s[3] <= 'f') ? s[3] - 'a' + 10 : (s[3] >= 'A' && s[3] <= 'F') ? s[3] - 'A' + 10 : -1;
            int lo = (s[4] >= '0' && s[4] <= '9') ? s[4] - '0' : (s[4] >= 'a' && s[4] <= 'f') ? s[4] - 'a' + 10 : (s[4] >= 'A' && s[4] <= 'F') ? s[4] - 'A' + 10 : -1;
            if (hi >= 0 && lo >= 0) {
                int b = hi * 16 + lo;
                t->byte_to_id[b] = i;
                if (t->types[i] == TOK_TYPE_NORMAL) t->types[i] = TOK_TYPE_BYTE;
            }
        }
    }

    /* Special token ids. */
    t->bos_id  = (int)pickle_meta_int(m, "tokenizer.ggml.bos_token_id",     -1);
    t->eos_id  = (int)pickle_meta_int(m, "tokenizer.ggml.eos_token_id",     -1);
    t->unk_id  = (int)pickle_meta_int(m, "tokenizer.ggml.unknown_token_id", -1);
    t->pad_id  = (int)pickle_meta_int(m, "tokenizer.ggml.padding_token_id", -1);
    t->add_bos = (int)pickle_meta_int(m, "tokenizer.ggml.add_bos_token",     0);
    t->add_eos = (int)pickle_meta_int(m, "tokenizer.ggml.add_eos_token",     0);

    *out_tok = t;
    return PICKLE_OK;
}

void pickle_tok_free(pickle_tokenizer_t* t) {
    if (!t) return;
    if (t->tokens) {
        for (int i = 0; i < t->vocab_size; i++) free(t->tokens[i]);
        free(t->tokens);
    }
    free(t->scores);
    free(t->types);
    free(t->byte_to_id);
    free(t->hash_keys);
    free(t->hash_vals);
    free(t);
}

int pickle_tok_id(const pickle_tokenizer_t* t, const char* s) {
    if (!t || !s) return -1;
    return tok_hash_find(t, s, strlen(s));
}

const char* pickle_tok_text(const pickle_tokenizer_t* t, int32_t id) {
    if (!t || id < 0 || id >= t->vocab_size) return 0;
    return t->tokens[id];
}

int pickle_tok_size(const pickle_tokenizer_t* t) {
    return t ? t->vocab_size : 0;
}
int pickle_tok_bos(const pickle_tokenizer_t* t) { return t ? t->bos_id : -1; }
int pickle_tok_eos(const pickle_tokenizer_t* t) { return t ? t->eos_id : -1; }
int pickle_tok_unk(const pickle_tokenizer_t* t) { return t ? t->unk_id : -1; }
int pickle_tok_pad(const pickle_tokenizer_t* t) { return t ? t->pad_id : -1; }

/* ------------------------------------------------------------------ */
/* Encode                                                             */
/* ------------------------------------------------------------------ */
/* Replace the SentencePiece space marker "▁" (U+2581) with an actual
 * space — but we encode in reverse: take the input text, split into
 * UTF-8 "words" on whitespace, prepend "▁" to each, then BPE-merge. */

/* Returns the byte length of the next UTF-8 char starting at p, or 0
 * if p points at '\0'. */
static size_t utf8_len(const unsigned char* p) {
    if (!p || *p == 0) return 0;
    if ((*p & 0x80) == 0) return 1;
    if ((*p & 0xE0) == 0xC0) return 2;
    if ((*p & 0xF0) == 0xE0) return 3;
    if ((*p & 0xF8) == 0xF0) return 4;
    return 1;   /* invalid — treat as single byte */
}

/* Convert text → token ids. Algorithm:
 *   1. For each "word" (run of non-whitespace UTF-8 chars), prefix
 *      with "▁" (U+2581).
 *   2. Decompose the word into single-char "symbols". Each symbol is
 *      a UTF-8 char (or a byte if invalid UTF-8). For each symbol,
 *      look up its token id; if not found and byte-fallback is
 *      available, emit the byte token.
 *   3. Greedily merge the highest-scoring adjacent pair until no
 *      mergeable pair remains.
 */
int pickle_tok_encode(pickle_tokenizer_t* t,
                      const char* text, int add_bos,
                      int32_t** out_ids, size_t* out_n) {
    if (!t || !text || !out_ids || !out_n) return PICKLE_ERR_ARG;
    *out_ids = 0; *out_n = 0;

    /* Output buffer — grown dynamically. */
    size_t cap = 64;
    size_t n = 0;
    int32_t* ids = (int32_t*)malloc(cap * sizeof(int32_t));
    if (!ids) return PICKLE_ERR_MEMORY;

    if (add_bos && t->bos_id >= 0) {
        ids[n++] = t->bos_id;
    }

    const unsigned char* p = (const unsigned char*)text;
    /* Walk the text, splitting on ASCII whitespace (matches llama.cpp's
     * default pre-tokenizer). */
    while (*p) {
        /* Skip leading whitespace. */
        while (*p && (*p == ' ' || *p == '\t' || *p == '\n' || *p == '\r')) p++;
        if (!*p) break;

        /* Collect a word: run of non-whitespace UTF-8 chars. We need
         * the word as bytes, then prepend "▁". */
        /* First, snapshot the word's start. */
        const unsigned char* word_start = p;
        while (*p && !(*p == ' ' || *p == '\t' || *p == '\n' || *p == '\r')) {
            size_t cl = utf8_len(p);
            if (cl == 0) break;
            p += cl;
        }
        size_t word_len = (size_t)(p - word_start);

        /* Build the SentencePiece "▁" + word string. */
        size_t buf_len = 3 + word_len + 1;   /* ▁ is 3 bytes in UTF-8 */
        char* word = (char*)malloc(buf_len);
        if (!word) { free(ids); return PICKLE_ERR_MEMORY; }
        word[0] = (char)0xE2; word[1] = (char)0x96; word[2] = (char)0x81;   /* U+2581 ▁ */
        memcpy(word + 3, word_start, word_len);
        word[buf_len - 1] = 0;

        /* Tokenise this word with BPE. */
        /* Initial symbol list: one symbol per UTF-8 char. */
        size_t sym_cap = 32;
        size_t sym_n = 0;
        /* Each symbol is (start, length) into word, plus its token id. */
        int* sym_id = (int*)malloc(sym_cap * sizeof(int));
        size_t* sym_off = (size_t*)malloc(sym_cap * sizeof(size_t));
        size_t* sym_len = (size_t*)malloc(sym_cap * sizeof(size_t));
        if (!sym_id || !sym_off || !sym_len) {
            free(word); free(sym_id); free(sym_off); free(sym_len); free(ids);
            return PICKLE_ERR_MEMORY;
        }

        size_t i = 0;
        while (i < buf_len - 1) {
            size_t cl = utf8_len((const unsigned char*)word + i);
            if (cl == 0 || i + cl > buf_len - 1) cl = 1;
            /* Look up this single-char piece. */
            int id = tok_hash_find(t, word + i, cl);
            if (sym_n == sym_cap) {
                sym_cap *= 2;
                sym_id  = (int*)realloc(sym_id,  sym_cap * sizeof(int));
                sym_off = (size_t*)realloc(sym_off, sym_cap * sizeof(size_t));
                sym_len = (size_t*)realloc(sym_len, sym_cap * sizeof(size_t));
                if (!sym_id || !sym_off || !sym_len) {
                    free(word); free(sym_id); free(sym_off); free(sym_len); free(ids);
                    return PICKLE_ERR_MEMORY;
                }
            }
            sym_off[sym_n] = i;
            sym_len[sym_n] = cl;
            sym_id [sym_n] = id;
            sym_n++;
            i += cl;
        }

        /* Greedily merge the highest-scoring adjacent pair. */
        while (sym_n > 1) {
            float best_score = -1e30f;
            int    best_id   = -1;
            size_t best_idx  = 0;
            for (size_t k = 0; k + 1 < sym_n; k++) {
                /* Concatenate sym[k] + sym[k+1] and look up. */
                size_t total = sym_len[k] + sym_len[k+1];
                /* Small enough for stack. */
                char buf[64];
                if (total > sizeof(buf) - 1) {
                    /* Too long — definitely not a token. Skip. */
                    continue;
                }
                memcpy(buf, word + sym_off[k], sym_len[k]);
                memcpy(buf + sym_len[k], word + sym_off[k+1], sym_len[k+1]);
                int id = tok_hash_find(t, buf, total);
                if (id < 0) continue;
                /* Only merge if both pieces are normal tokens (not
                 * special, not byte fallback). */
                if (t->types[id] == TOK_TYPE_CONTROL ||
                    t->types[id] == TOK_TYPE_USER_DEFINED) continue;
                if (t->types[sym_id[k]] == TOK_TYPE_CONTROL ||
                    t->types[sym_id[k]] == TOK_TYPE_USER_DEFINED ||
                    t->types[sym_id[k+1]] == TOK_TYPE_CONTROL ||
                    t->types[sym_id[k+1]] == TOK_TYPE_USER_DEFINED) continue;
                float sc = t->scores[id];
                if (sc > best_score) {
                    best_score = sc;
                    best_id = id;
                    best_idx = k;
                }
            }
            if (best_id < 0) break;   /* no more merges */
            /* Apply the merge: sym[best_idx] becomes the merged symbol,
             * sym[best_idx+1] is removed. */
            sym_id [best_idx] = best_id;
            sym_len[best_idx] += sym_len[best_idx + 1];
            for (size_t k = best_idx + 1; k + 1 < sym_n; k++) {
                sym_id [k] = sym_id [k+1];
                sym_off[k] = sym_off[k+1];
                sym_len[k] = sym_len[k+1];
            }
            sym_n--;
        }

        /* Emit symbols, falling back to byte tokens for unknowns. */
        for (size_t k = 0; k < sym_n; k++) {
            int id = sym_id[k];
            if (id < 0) {
                /* Byte-fallback: emit one <0xXX> per byte. */
                for (size_t b = 0; b < sym_len[k]; b++) {
                    unsigned char byte = (unsigned char)word[sym_off[k] + b];
                    int bid = t->byte_to_id[byte];
                    if (bid < 0) bid = (t->unk_id >= 0) ? t->unk_id : 0;
                    if (n == cap) {
                        cap *= 2;
                        ids = (int32_t*)realloc(ids, cap * sizeof(int32_t));
                        if (!ids) { free(word); free(sym_id); free(sym_off); free(sym_len); return PICKLE_ERR_MEMORY; }
                    }
                    ids[n++] = bid;
                }
            } else {
                if (n == cap) {
                    cap *= 2;
                    ids = (int32_t*)realloc(ids, cap * sizeof(int32_t));
                    if (!ids) { free(word); free(sym_id); free(sym_off); free(sym_len); return PICKLE_ERR_MEMORY; }
                }
                ids[n++] = id;
            }
        }

        free(word); free(sym_id); free(sym_off); free(sym_len);
    }

    *out_ids = ids;
    *out_n = n;
    return PICKLE_OK;
}

/* ------------------------------------------------------------------ */
/* Decode                                                             */
/* ------------------------------------------------------------------ */
int pickle_tok_decode(pickle_tokenizer_t* t,
                      const int32_t* ids, size_t n,
                      int skip_special,
                      char** out_text) {
    if (!t || !ids || !out_text) return PICKLE_ERR_ARG;
    *out_text = 0;

    /* Build the output string in a growable buffer. */
    size_t cap = 256;
    size_t len = 0;
    char* out = (char*)malloc(cap);
    if (!out) return PICKLE_ERR_MEMORY;

    for (size_t i = 0; i < n; i++) {
        int id = ids[i];
        if (id < 0 || id >= t->vocab_size) continue;
        const char* s = t->tokens[id];
        size_t slen = strlen(s);

        /* Byte-fallback token "<0xXX>" — emit the raw byte. */
        if (t->types[id] == TOK_TYPE_BYTE ||
            (slen == 6 && s[0] == '<' && s[1] == '0' && s[2] == 'x' && s[5] == '>')) {
            int hi = (s[3] >= '0' && s[3] <= '9') ? s[3] - '0' : (s[3] >= 'a' && s[3] <= 'f') ? s[3] - 'a' + 10 : (s[3] >= 'A' && s[3] <= 'F') ? s[3] - 'A' + 10 : -1;
            int lo = (s[4] >= '0' && s[4] <= '9') ? s[4] - '0' : (s[4] >= 'a' && s[4] <= 'f') ? s[4] - 'a' + 10 : (s[4] >= 'A' && s[4] <= 'F') ? s[4] - 'A' + 10 : -1;
            if (hi >= 0 && lo >= 0) {
                unsigned char b = (unsigned char)(hi * 16 + lo);
                if (len + 1 >= cap) {
                    cap *= 2;
                    out = (char*)realloc(out, cap);
                    if (!out) return PICKLE_ERR_MEMORY;
                }
                out[len++] = (char)b;
                continue;
            }
        }

        /* Special token (begins with '<' and ends with '>') — skip if
         * asked, else emit literally. */
        if (skip_special &&
            (t->types[id] == TOK_TYPE_CONTROL ||
             t->types[id] == TOK_TYPE_USER_DEFINED)) {
            continue;
        }
        /* Also skip BOS/EOS/PAD/UNK ids if asked. */
        if (skip_special &&
            (id == t->bos_id || id == t->eos_id ||
             id == t->pad_id || id == t->unk_id)) {
            continue;
        }

        /* Replace "▁" (U+2581, 3 bytes E2 96 81) with space. */
        for (size_t j = 0; j < slen; j++) {
            if ((unsigned char)s[j] == 0xE2 &&
                j + 2 < slen &&
                (unsigned char)s[j+1] == 0x96 &&
                (unsigned char)s[j+2] == 0x81) {
                if (len + 1 >= cap) {
                    cap *= 2;
                    out = (char*)realloc(out, cap);
                    if (!out) return PICKLE_ERR_MEMORY;
                }
                out[len++] = ' ';
                j += 2;
            } else {
                if (len + 1 >= cap) {
                    cap *= 2;
                    out = (char*)realloc(out, cap);
                    if (!out) return PICKLE_ERR_MEMORY;
                }
                out[len++] = s[j];
            }
        }
    }

    if (len + 1 > cap) {
        cap = len + 1;
        out = (char*)realloc(out, cap);
        if (!out) return PICKLE_ERR_MEMORY;
    }
    out[len] = 0;
    *out_text = out;
    return PICKLE_OK;
}
