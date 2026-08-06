/*
 * pickle_cli.c — host CLI frontend for the pickle GGUF engine.
 * Copyright (c) 2026 lestramk.org / Lee Muriihi Kingori
 *
 * Subcommands:
 *
 *   pickle selftest
 *       Runs pickle_selftest() on the embedded demo GGUF. Should print
 *       "pickle: selftest OK, next token = N".
 *
 *   pickle info <model.gguf>
 *       Loads metadata only, prints the detected architecture (arch,
 *       n_layers, hidden_dim, n_heads, n_kv_heads, head_dim, vocab_size,
 *       max_seq_len, rope_theta, rms_eps, tensor_count) and lists the
 *       first 12 tensors with their shapes and types.
 *
 *   pickle infer <model.gguf> "<prompt>" [N] [--temp T] [--top-p P] [--seed S]
 *       Fast-path inference: tokenises the prompt with the model's BPE
 *       tokenizer, runs the Llama forward pass, and greedily generates
 *       N tokens (default 20). Tokens are streamed to stdout as soon as
 *       they are produced (detokenised on the fly). With --temp > 0,
 *       uses temperature + top_p sampling.
 *
 *   pickle chat <model.gguf> [--temp T] [--top-p P] [--seed S] [--max N]
 *       Interactive REPL: reads prompts from stdin, streams responses.
 *       Type :quit to exit, :reset to clear the conversation (currently
 *       each prompt is independent — multi-turn context is a v0.5 item).
 *
 *   pickle bench <model.gguf> ["<prompt>"] [N] [--temp T]
 *       Benchmarks inference: prints prompt tokens/sec (prefill) and
 *       generated tokens/sec (decode). Default N=64, default prompt is
 *       "Hello, my name is".
 *
 *   pickle tokens <model.gguf> <encode|decode> <text|ids...>
 *       Encodes text to token ids, or decodes token ids to text.
 *         pickle tokens model.gguf encode "Hello world"
 *         pickle tokens model.gguf decode 1 1505 29892 1599
 *
 *   pickle dequant <model.gguf> <tensor_name>
 *       Prints the first 32 float values of the named tensor after
 *       dequantization (legacy — uses the metadata-only load + on-
 *       demand dequant path).
 *
 * Exit codes: 0 on success, 1 on error.
 */
#include "pickle.h"
#include "pickle_fast.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <ctype.h>

/* Forward declarations from pickle_host.c */
int  pickle_load_from_file      (const char* path, pickle_model_t** out_model);
int  pickle_load_from_file_meta (const char* path, pickle_model_t** out_model);
int  pickle_load_from_file_lazy (const char* path, pickle_model_t** out_model);
int  pickle_load_from_file_mmap (const char* path, pickle_model_t** out_model);
int  pickle_run_prompt(pickle_model_t* model, const pickle_arch_t* arch,
                       const char* prompt_str, size_t n_gen,
                       int32_t* out_buf, size_t out_buf_size, size_t* out_n);
int  pickle_dequant_tensor(pickle_model_t* m, size_t idx);

int  pickle_chat_loop(
    pickle_model_t*        model,
    const pickle_arch_t*   arch,
    pickle_tokenizer_t*    tok,
    const char*            prompt,
    int                    add_bos,
    size_t                 max_new_tokens,
    float                  temp,
    float                  top_p,
    uint32_t               seed,
    int                    (*on_token)(int32_t id, void* user_ctx),
    void*                  user_ctx,
    size_t*                out_prompt_n,
    size_t*                out_generated_n,
    double*                out_prefill_ms,
    double*                out_decode_avg_ms);

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
        "  pickle selftest                                 Run the embedded selftest\n"
        "  pickle info <model.gguf>                        Print model architecture + tensors\n"
        "  pickle infer <model.gguf> \"<prompt>\" [N] [opts]  Generate N tokens (default 20)\n"
        "  pickle chat  <model.gguf> [opts]                Interactive REPL (streaming)\n"
        "  pickle bench <model.gguf> [\"prompt\"] [N] [opts] Benchmark prefill + decode tok/s\n"
        "  pickle tokens <model.gguf> encode \"<text>\"      BPE-encode text → token ids\n"
        "  pickle tokens <model.gguf> decode <id...>        Decode token ids → text\n"
        "  pickle dequant <model.gguf> <tensor>            Print first 32 floats of <tensor>\n"
        "\n"
        "Options (infer / chat / bench):\n"
        "  --temp T     Temperature (default 0 = greedy)\n"
        "  --top-p P    Nucleus sampling threshold (default 0.9)\n"
        "  --seed S     RNG seed (default 0xC0FFEE)\n"
        "  --max N      Max new tokens (default 20 infer / 256 chat / 64 bench)\n"
        "  --no-bos     Don't prepend BOS token\n");
}

/* Parse a long option of the form "--name value" from argv. Returns
 * 1 if consumed (and *out_val set), 0 if not recognised. */
static int parse_opt(int argc, char** argv, int* i, const char* name, const char** out_val) {
    if (strcmp(argv[*i], name) == 0) {
        if (*i + 1 >= argc) {
            fprintf(stderr, "pickle: %s requires a value\n", name);
            return -1;
        }
        *out_val = argv[*i + 1];
        *i += 2;
        return 1;
    }
    return 0;
}

typedef struct {
    float   temp;
    float   top_p;
    uint32_t seed;
    int     max_new;
    int     add_bos;
} cli_opts_t;

static void opts_default(cli_opts_t* o, int default_max) {
    o->temp = 0.0f;
    o->top_p = 0.9f;
    o->seed = 0xC0FFEEu;
    o->max_new = default_max;
    o->add_bos = 1;
}

/* Consume any --temp / --top-p / --seed / --max / --no-bos options
 * from argv starting at *i. Returns 0 on success, -1 on error. */
static int consume_opts(int argc, char** argv, int* i, cli_opts_t* o) {
    while (*i < argc) {
        const char* v;
        int r;
        if ((r = parse_opt(argc, argv, i, "--temp", &v)))   { if (r < 0) return -1; o->temp = (float)atof(v); continue; }
        if ((r = parse_opt(argc, argv, i, "--top-p", &v)))  { if (r < 0) return -1; o->top_p = (float)atof(v); continue; }
        if ((r = parse_opt(argc, argv, i, "--seed", &v)))   { if (r < 0) return -1; o->seed = (uint32_t)strtoul(v, 0, 0); continue; }
        if ((r = parse_opt(argc, argv, i, "--max", &v)))    { if (r < 0) return -1; o->max_new = atoi(v); continue; }
        if (strcmp(argv[*i], "--no-bos") == 0)              { o->add_bos = 0; (*i)++; continue; }
        /* Not an option — stop. */
        break;
    }
    return 0;
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
    int rc = pickle_load_from_file_meta(path, &m);
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

    /* Print tensor dtype distribution + total on-disk size. */
    size_t total_bytes = 0;
    int type_counts[16] = {0};
    for (size_t j = 0; j < tc; j++) {
        pickle_tensor_info_t ti;
        if (pickle_tensor_info(m, j, &ti) != PICKLE_OK) break;
        if (ti.type < 16) type_counts[ti.type]++;
        total_bytes += ti.data_size;
    }
    printf("total_tensor_bytes = %zu  (%.1f MiB)\n", total_bytes, total_bytes / 1048576.0);

    size_t list_n = tc < 16 ? tc : 16;
    printf("--- first %zu tensors ---\n", list_n);
    for (size_t i = 0; i < list_n; i++) {
        pickle_tensor_info_t ti;
        if (pickle_tensor_info(m, i, &ti) != PICKLE_OK) break;
        printf("  [%3zu] %-32s %s  dims=[", i, ti.name, dtype_name(ti.type));
        for (uint32_t d = 0; d < ti.n_dims; d++) {
            printf("%llu", (unsigned long long)ti.dims[d]);
            if (d + 1 < ti.n_dims) printf(",");
        }
        printf("] n_elems=%llu  bytes=%zu\n",
               (unsigned long long)ti.n_elements, ti.data_size);
    }

    /* Try to detect a tokenizer. */
    pickle_tokenizer_t* tok = 0;
    if (pickle_tok_init(m, &tok) == PICKLE_OK) {
        printf("\ntokenizer:\n");
        printf("  vocab_size    = %d\n", pickle_tok_size(tok));
        printf("  bos_id        = %d\n", pickle_tok_bos(tok));
        printf("  eos_id        = %d\n", pickle_tok_eos(tok));
        printf("  unk_id        = %d\n", pickle_tok_unk(tok));
        printf("  pad_id        = %d\n", pickle_tok_pad(tok));
        /* Show first 5 tokens as a sanity check. */
        printf("  first tokens  = ");
        for (int j = 0; j < 5 && j < pickle_tok_size(tok); j++) {
            const char* s = pickle_tok_text(tok, j);
            printf("[%d:\"", j);
            /* Print safely — escape newlines etc. */
            for (const char* p = s; p && *p; p++) {
                if (*p == '\n') printf("\\n");
                else if (*p == '\t') printf("\\t");
                else if ((unsigned char)*p < 0x20) printf("\\x%02x", (unsigned char)*p);
                else putchar(*p);
            }
            printf("\"] ");
        }
        printf("\n");
        pickle_tok_free(tok);
    } else {
        printf("\ntokenizer: none (not a Llama BPE model?)\n");
    }

    pickle_free(m);
    return 0;
}

/* ---- infer / chat / bench all use the streaming callback ---------- */
typedef struct {
    pickle_tokenizer_t* tok;
    int                 first;
    int                 print_ids;   /* for infer: also print token ids */
    FILE*               out;
} stream_ctx_t;

static int stream_token(int32_t id, void* user_ctx) {
    stream_ctx_t* c = (stream_ctx_t*)user_ctx;
    /* Decode this single token to text. */
    int32_t one = id;
    char* text = NULL;
    int rc = pickle_tok_decode(c->tok, &one, 1, 0 /*keep special*/, &text);
    if (rc == PICKLE_OK && text) {
        fputs(text, c->out);
        free(text);
    }
    fflush(c->out);
    c->first = 0;
    return 0;
}

static int cmd_infer(const char* path, const char* prompt, cli_opts_t* opts) {
    pickle_model_t* m = 0;
    int rc = pickle_load_from_file_mmap(path, &m);
    if (rc != PICKLE_OK) return 1;

    pickle_arch_t a;
    rc = pickle_arch_detect(m, &a);
    if (rc != PICKLE_OK) {
        fprintf(stderr, "pickle: arch_detect failed: rc=%d\n", rc);
        pickle_free(m);
        return 1;
    }

    pickle_tokenizer_t* tok = 0;
    rc = pickle_tok_init(m, &tok);
    if (rc != PICKLE_OK) {
        fprintf(stderr, "pickle: tokenizer init failed: rc=%d\n", rc);
        pickle_free(m);
        return 1;
    }

    stream_ctx_t ctx = { tok, 1, 0, stdout };
    size_t prompt_n = 0, gen_n = 0;
    double prefill_ms = 0, decode_avg_ms = 0;
    rc = pickle_chat_loop(m, &a, tok, prompt, opts->add_bos,
                          (size_t)opts->max_new,
                          opts->temp, opts->top_p, opts->seed,
                          stream_token, &ctx,
                          &prompt_n, &gen_n, &prefill_ms, &decode_avg_ms);
    printf("\n--- %zu prompt tokens, %zu generated in %.1f ms prefill + %.1f ms/tok decode ---\n",
           prompt_n, gen_n, prefill_ms, decode_avg_ms);

    pickle_tok_free(tok);
    pickle_free(m);
    return rc == PICKLE_OK ? 0 : 1;
}

static int cmd_bench(const char* path, const char* prompt, cli_opts_t* opts) {
    pickle_model_t* m = 0;
    int rc = pickle_load_from_file_mmap(path, &m);
    if (rc != PICKLE_OK) return 1;

    pickle_arch_t a;
    rc = pickle_arch_detect(m, &a);
    if (rc != PICKLE_OK) {
        fprintf(stderr, "pickle: arch_detect failed: rc=%d\n", rc);
        pickle_free(m);
        return 1;
    }

    pickle_tokenizer_t* tok = 0;
    rc = pickle_tok_init(m, &tok);
    if (rc != PICKLE_OK) {
        fprintf(stderr, "pickle: tokenizer init failed: rc=%d\n", rc);
        pickle_free(m);
        return 1;
    }

    stream_ctx_t ctx = { tok, 1, 0, stdout };
    size_t prompt_n = 0, gen_n = 0;
    double prefill_ms = 0, decode_avg_ms = 0;
    /* Use a null callback (we still want to know it ran). Actually we
     * want to discard the output for clean benchmark numbers, so pass
     * a no-op callback. */
    rc = pickle_chat_loop(m, &a, tok, prompt, opts->add_bos,
                          (size_t)opts->max_new,
                          opts->temp, opts->top_p, opts->seed,
                          NULL, NULL,
                          &prompt_n, &gen_n, &prefill_ms, &decode_avg_ms);

    double prefill_tok_s = prefill_ms > 0 ? (double)prompt_n / (prefill_ms / 1000.0) : 0;
    double decode_tok_s   = decode_avg_ms > 0 ? 1000.0 / decode_avg_ms : 0;

    printf("\n");
    printf("model         : %s\n", path);
    printf("arch          : %s  L=%d H=%d HK=%d D=%d HD=%d VS=%d\n",
           a.arch_name, a.n_layers, a.n_heads, a.n_kv_heads,
           a.head_dim, a.hidden_dim, a.vocab_size);
    printf("prompt tokens : %zu  (prefill in %.1f ms = %.1f tok/s)\n",
           prompt_n, prefill_ms, prefill_tok_s);
    printf("generated     : %zu  (avg %.2f ms/tok = %.1f tok/s decode)\n",
           gen_n, decode_avg_ms, decode_tok_s);

    pickle_tok_free(tok);
    pickle_free(m);
    return rc == PICKLE_OK ? 0 : 1;
}

static int cmd_chat(const char* path, cli_opts_t* opts) {
    pickle_model_t* m = 0;
    int rc = pickle_load_from_file_mmap(path, &m);
    if (rc != PICKLE_OK) return 1;

    pickle_arch_t a;
    rc = pickle_arch_detect(m, &a);
    if (rc != PICKLE_OK) {
        fprintf(stderr, "pickle: arch_detect failed: rc=%d\n", rc);
        pickle_free(m);
        return 1;
    }

    pickle_tokenizer_t* tok = 0;
    rc = pickle_tok_init(m, &tok);
    if (rc != PICKLE_OK) {
        fprintf(stderr, "pickle: tokenizer init failed: rc=%d\n", rc);
        pickle_free(m);
        return 1;
    }

    fprintf(stderr, "pickle chat — model: %s (%s, %dL). Type :quit to exit.\n",
            path, a.arch_name, a.n_layers);

    char buf[8192];
    while (1) {
        fprintf(stderr, "\n>>> ");
        if (!fgets(buf, sizeof(buf), stdin)) break;
        size_t l = strlen(buf);
        while (l > 0 && (buf[l-1] == '\n' || buf[l-1] == '\r')) buf[--l] = 0;
        if (l == 0) continue;
        if (strcmp(buf, ":quit") == 0 || strcmp(buf, ":q") == 0) break;
        if (strcmp(buf, ":reset") == 0) continue;  /* no-op for now */

        stream_ctx_t ctx = { tok, 1, 0, stdout };
        size_t prompt_n = 0, gen_n = 0;
        double prefill_ms = 0, decode_avg_ms = 0;
        rc = pickle_chat_loop(m, &a, tok, buf, opts->add_bos,
                              (size_t)opts->max_new,
                              opts->temp, opts->top_p, opts->seed,
                              stream_token, &ctx,
                              &prompt_n, &gen_n, &prefill_ms, &decode_avg_ms);
        if (rc != PICKLE_OK) {
            fprintf(stderr, "\npickle: chat_loop failed rc=%d\n", rc);
        }
        printf("\n");
    }

    pickle_tok_free(tok);
    pickle_free(m);
    return 0;
}

static int cmd_tokens(const char* path, const char* subcmd, int argc, char** argv) {
    pickle_model_t* m = 0;
    int rc = pickle_load_from_file_meta(path, &m);
    if (rc != PICKLE_OK) return 1;

    pickle_tokenizer_t* tok = 0;
    rc = pickle_tok_init(m, &tok);
    if (rc != PICKLE_OK) {
        fprintf(stderr, "pickle: tokenizer init failed: rc=%d\n", rc);
        pickle_free(m);
        return 1;
    }

    if (strcmp(subcmd, "encode") == 0) {
        if (argc < 1) {
            fprintf(stderr, "pickle: tokens encode needs text\n");
            pickle_tok_free(tok); pickle_free(m);
            return 1;
        }
        const char* text = argv[0];
        int32_t* ids = NULL;
        size_t n = 0;
        rc = pickle_tok_encode(tok, text, 0 /*no BOS for inspection*/, &ids, &n);
        if (rc != PICKLE_OK) {
            fprintf(stderr, "pickle: encode failed rc=%d\n", rc);
            pickle_tok_free(tok); pickle_free(m);
            return 1;
        }
        printf("%zu tokens:\n", n);
        for (size_t i = 0; i < n; i++) {
            const char* s = pickle_tok_text(tok, ids[i]);
            printf("  [%3zu] id=%-8d ", i, ids[i]);
            if (s) {
                putchar('"');
                for (const char* p = s; *p; p++) {
                    if (*p == '\n') printf("\\n");
                    else if (*p == '\t') printf("\\t");
                    else if ((unsigned char)*p < 0x20) printf("\\x%02x", (unsigned char)*p);
                    else putchar(*p);
                }
                putchar('"');
            }
            printf("\n");
        }
        free(ids);
    } else if (strcmp(subcmd, "decode") == 0) {
        int32_t* ids = (int32_t*)malloc(argc * sizeof(int32_t));
        if (!ids) { pickle_tok_free(tok); pickle_free(m); return 1; }
        size_t n = 0;
        for (int i = 0; i < argc; i++) {
            ids[n] = (int32_t)strtol(argv[i], 0, 10);
            if (ids[n] < 0 || ids[n] >= pickle_tok_size(tok)) {
                fprintf(stderr, "pickle: token id %d out of range\n", ids[n]);
                continue;
            }
            n++;
        }
        char* text = NULL;
        rc = pickle_tok_decode(tok, ids, n, 0, &text);
        if (rc == PICKLE_OK && text) {
            printf("%s\n", text);
            free(text);
        }
        free(ids);
    } else {
        fprintf(stderr, "pickle: unknown tokens subcommand '%s'\n", subcmd);
        pickle_tok_free(tok); pickle_free(m);
        return 1;
    }

    pickle_tok_free(tok);
    pickle_free(m);
    return 0;
}

static int cmd_dequant(const char* path, const char* tensor_name) {
    pickle_model_t* m = 0;
    int rc = pickle_load_from_file_meta(path, &m);
    if (rc != PICKLE_OK) return 1;

    int ti = pickle_tensor_find(m, tensor_name);
    if (ti < 0) {
        fprintf(stderr, "pickle: tensor '%s' not found in '%s'\n",
                tensor_name, path);
        pickle_free(m);
        return 1;
    }

    rc = pickle_dequant_tensor(m, (size_t)ti);
    if (rc != PICKLE_OK) {
        fprintf(stderr, "pickle: dequant_tensor failed: rc=%d\n", rc);
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
        if (argc < 4) { usage(stderr); return 1; }
        const char* path = argv[2];
        const char* prompt = argv[3];
        cli_opts_t opts; opts_default(&opts, 20);
        int i = 4;
        if (i < argc && argv[i][0] >= '0' && argv[i][0] <= '9') {
            opts.max_new = atoi(argv[i]);
            i++;
        }
        if (consume_opts(argc, argv, &i, &opts) < 0) return 1;
        return cmd_infer(path, prompt, &opts);
    }
    if (strcmp(cmd, "bench") == 0) {
        if (argc < 3) { usage(stderr); return 1; }
        const char* path = argv[2];
        const char* prompt = "Hello, my name is";
        cli_opts_t opts; opts_default(&opts, 64);
        int i = 3;
        if (i < argc && argv[i][0] != '-' && !isdigit((unsigned char)argv[i][0])) {
            /* Treat as prompt. */
            prompt = argv[i]; i++;
        }
        if (i < argc && argv[i][0] >= '0' && argv[i][0] <= '9') {
            opts.max_new = atoi(argv[i]); i++;
        }
        if (consume_opts(argc, argv, &i, &opts) < 0) return 1;
        return cmd_bench(path, prompt, &opts);
    }
    if (strcmp(cmd, "chat") == 0) {
        if (argc < 3) { usage(stderr); return 1; }
        const char* path = argv[2];
        cli_opts_t opts; opts_default(&opts, 256);
        int i = 3;
        if (consume_opts(argc, argv, &i, &opts) < 0) return 1;
        return cmd_chat(path, &opts);
    }
    if (strcmp(cmd, "tokens") == 0) {
        if (argc < 5) { usage(stderr); return 1; }
        const char* path = argv[2];
        const char* subcmd = argv[3];
        return cmd_tokens(path, subcmd, argc - 4, &argv[4]);
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
