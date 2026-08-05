/*
 * pickle_cli.c — host CLI frontend for the pickle GGUF engine.
 * Copyright (c) 2026 lestramk.org / Lee Muriihi Kingori
 *
 * Subcommands:
 *
 *   pickle selftest
 *       Runs pickle_selftest() on the embedded demo GGUF. Should print
 *       "pickle: selftest OK, next token = 6".
 *
 *   pickle info <model.gguf>
 *       Loads <model.gguf>, prints the detected architecture (arch,
 *       n_layers, hidden_dim, n_heads, n_kv_heads, head_dim, vocab_size,
 *       max_seq_len, rope_theta, rms_eps, tensor_count) and lists the
 *       first 12 tensors with their shapes and types.
 *
 *   pickle infer <model.gguf> "<prompt>" [N]
 *       Loads <model.gguf>, runs the Llama forward pass on <prompt>
 *       (character-level tokenized for the demo model), greedily
 *       generates N tokens (default N=20), and prints them as ASCII
 *       chars.
 *
 *   pickle dequant <model.gguf> <tensor_name>
 *       Prints the first 32 float values of the named tensor after
 *       dequantization (uses sfp_to_float on the stored sfp_t bit
 *       patterns).
 *
 * Exit codes: 0 on success, 1 on error.
 */
#include "pickle.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>

/* Forward declarations from pickle_host.c */
int  pickle_load_from_file(const char* path, pickle_model_t** out_model);
int  pickle_run_prompt(pickle_model_t* model, const pickle_arch_t* arch,
                       const char* prompt_str, size_t n_gen,
                       int32_t* out_buf, size_t out_buf_size, size_t* out_n);

/* ---- helpers ------------------------------------------------------- */
static const char* dtype_name(uint32_t t) {
    switch (t) {
        case GGML_F32:  return "F32";
        case GGML_F16:  return "F16";
        case GGML_Q4_0: return "Q4_0";
        case GGML_Q4_1: return "Q4_1";
        case GGML_Q5_0: return "Q5_0";
        case GGML_Q5_1: return "Q5_1";
        case GGML_Q8_0: return "Q8_0";
        case GGML_Q8_1: return "Q8_1";
        case GGML_Q2_K: return "Q2_K";
        case GGML_Q3_K: return "Q3_K";
        case GGML_Q4_K: return "Q4_K";
        case GGML_Q5_K: return "Q5_K";
        case GGML_Q6_K: return "Q6_K";
        case GGML_Q8_K: return "Q8_K";
        default:        return "??";
    }
}

static void usage(FILE* f) {
    fprintf(f,
        "pickle — a from-scratch GGUF model loader and inference engine\n"
        "\n"
        "Usage:\n"
        "  pickle selftest                         Run the embedded selftest\n"
        "  pickle info <model.gguf>                Print model architecture + tensors\n"
        "  pickle infer <model.gguf> \"<prompt>\" [N] Generate N tokens (default 20)\n"
        "  pickle dequant <model.gguf> <tensor>    Print first 32 floats of <tensor>\n");
}

/* ---- subcommands --------------------------------------------------- */
static int cmd_selftest(void) {
    int32_t tok = -1;
    int rc = pickle_selftest(&tok);
    if (rc != PICKLE_OK) {
        fprintf(stderr, "pickle: selftest FAILED: rc=%d\n", rc);
        return 1;
    }
    printf("selftest rc=0 token=%d\n", (int)tok);
    return 0;
}

static int cmd_info(const char* path) {
    pickle_model_t* m = 0;
    int rc = pickle_load_from_file(path, &m);
    if (rc != PICKLE_OK) return 1;

    pickle_arch_t a;
    rc = pickle_arch_detect(m, &a);
    if (rc != PICKLE_OK) {
        fprintf(stderr, "pickle: arch_detect failed: rc=%d\n", rc);
        pickle_free(m);
        return 1;
    }

    printf("arch             = %s\n", a.arch_name);
    printf("n_layers         = %d\n", a.n_layers);
    printf("hidden_dim       = %d\n", a.hidden_dim);
    printf("n_heads          = %d\n", a.n_heads);
    printf("n_kv_heads       = %d\n", a.n_kv_heads);
    printf("head_dim         = %d\n", a.head_dim);
    printf("intermediate_dim = %d\n", a.intermediate_dim);
    printf("vocab_size       = %d\n", a.vocab_size);
    printf("max_seq_len      = %d\n", a.max_seq_len);
    printf("rope_theta       = %g\n",   sfp_to_float(a.rope_theta_bits));
    printf("rope_freq_scale  = %g\n",   sfp_to_float(a.rope_freq_scale_bits));
    printf("rms_eps          = %g\n",   sfp_to_float(a.rms_eps_bits));
    printf("tie_word_embeds  = %d\n",   a.tie_word_embeddings);
    printf("norm_type        = %d\n",   a.norm_type);
    printf("act_type         = %d\n",   a.act_type);
    printf("rope_type        = %d\n",   a.rope_type);

    size_t tc = pickle_tensor_count(m);
    printf("tensor_count     = %zu\n", tc);

    size_t list_n = tc < 12 ? tc : 12;
    printf("--- first %zu tensors ---\n", list_n);
    for (size_t i = 0; i < list_n; i++) {
        pickle_tensor_info_t ti;
        if (pickle_tensor_info(m, i, &ti) != PICKLE_OK) break;
        printf("  [%2zu] %-28s %s  dims=[", i, ti.name, dtype_name(ti.type));
        for (uint32_t d = 0; d < ti.n_dims; d++) {
            printf("%llu", (unsigned long long)ti.dims[d]);
            if (d + 1 < ti.n_dims) printf(",");
        }
        printf("] n_elems=%llu\n", (unsigned long long)ti.n_elements);
    }

    pickle_free(m);
    return 0;
}

static int cmd_infer(const char* path, const char* prompt, long n_gen_in) {
    size_t n_gen = (n_gen_in <= 0) ? 20 : (size_t)n_gen_in;

    pickle_model_t* m = 0;
    int rc = pickle_load_from_file(path, &m);
    if (rc != PICKLE_OK) return 1;

    pickle_arch_t a;
    rc = pickle_arch_detect(m, &a);
    if (rc != PICKLE_OK) {
        fprintf(stderr, "pickle: arch_detect failed: rc=%d\n", rc);
        pickle_free(m);
        return 1;
    }

    int32_t* out = (int32_t*)calloc(n_gen, sizeof(int32_t));
    if (!out) {
        fprintf(stderr, "pickle: out of memory\n");
        pickle_free(m);
        return 1;
    }

    size_t got = 0;
    rc = pickle_run_prompt(m, &a, prompt, n_gen, out, n_gen, &got);
    if (rc != PICKLE_OK) {
        fprintf(stderr, "pickle: infer failed: rc=%d\n", rc);
        free(out);
        pickle_free(m);
        return 1;
    }

    printf("generated %zu tokens: ", got);
    /* Print tokens as ASCII chars (treating them as ASCII codes for the
     * demo model). Non-printable chars are emitted raw; that's fine for
     * the demo. */
    for (size_t i = 0; i < got; i++) {
        int32_t t = out[i];
        if (t < 0) continue;
        unsigned char c = (unsigned char)t;
        putchar((int)c);
    }
    putchar('\n');
    /* Also print the token IDs in brackets on a second line — the chars
     * above are often non-printable for the tiny demo model, so the IDs
     * make it easy to verify that N distinct tokens were actually
     * generated. */
    printf("  token ids: [");
    for (size_t i = 0; i < got; i++) {
        printf("%d", (int)out[i]);
        if (i + 1 < got) printf(", ");
    }
    printf("]\n");

    free(out);
    pickle_free(m);
    return 0;
}

static int cmd_dequant(const char* path, const char* tensor_name) {
    pickle_model_t* m = 0;
    int rc = pickle_load_from_file(path, &m);
    if (rc != PICKLE_OK) return 1;

    int ti = pickle_tensor_find(m, tensor_name);
    if (ti < 0) {
        fprintf(stderr, "pickle: tensor '%s' not found in '%s'\n",
                tensor_name, path);
        pickle_free(m);
        return 1;
    }

    pickle_tensor_info_t info;
    rc = pickle_tensor_info(m, (size_t)ti, &info);
    if (rc != PICKLE_OK) {
        fprintf(stderr, "pickle: tensor_info failed: rc=%d\n", rc);
        pickle_free(m);
        return 1;
    }

    printf("tensor: %s  type=%s  n_elements=%llu\n",
           info.name, dtype_name(info.type),
           (unsigned long long)info.n_elements);
    printf("first 32 values (dequantized to F32):\n");

    /* info.data is the dequantized F32 buffer (stored as sfp_t bit
     * patterns). Print the first 32 via sfp_to_float. */
    const sfp_t* p = (const sfp_t*)info.data;
    size_t n = 32;
    if (n > (size_t)info.n_elements) n = (size_t)info.n_elements;
    for (size_t i = 0; i < n; i++) {
        printf("  [%2zu] %g\n", i, sfp_to_float(p[i]));
    }

    pickle_free(m);
    return 0;
}

/* ---- main ---------------------------------------------------------- */
int main(int argc, char** argv) {
    if (argc < 2) {
        usage(stderr);
        return 1;
    }

    const char* cmd = argv[1];

    if (strcmp(cmd, "selftest") == 0) {
        return cmd_selftest();
    }
    if (strcmp(cmd, "info") == 0) {
        if (argc != 3) { usage(stderr); return 1; }
        return cmd_info(argv[2]);
    }
    if (strcmp(cmd, "infer") == 0) {
        if (argc != 4 && argc != 5) { usage(stderr); return 1; }
        long n = 20;
        if (argc == 5) n = strtol(argv[4], 0, 10);
        return cmd_infer(argv[2], argv[3], n);
    }
    if (strcmp(cmd, "dequant") == 0) {
        if (argc != 4) { usage(stderr); return 1; }
        return cmd_dequant(argv[2], argv[3]);
    }
    if (strcmp(cmd, "-h") == 0 || strcmp(cmd, "--help") == 0 ||
        strcmp(cmd, "help") == 0) {
        usage(stdout);
        return 0;
    }

    fprintf(stderr, "pickle: unknown command '%s'\n", cmd);
    usage(stderr);
    return 1;
}
