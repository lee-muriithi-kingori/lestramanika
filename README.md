# pickle — a from-scratch GGUF model loader and inference engine

> No llama.cpp. No ollama. No ggml. Pure C. Runs in the lestraOS kernel and on the host.

`pickle` is a tiny, self-contained GGUF (v3) parser + transformer
forward-pass engine written in freestanding C. It does **not** depend on
llama.cpp, ggml, ollama, or any other inference framework — every line of
the parser, dequantizer, tokenizer, and forward pass is in this repo. The
same source compiles two ways:

- **In the [lestraOS](https://github.com/lee-muriethi-kingori/LestraOS)
  kernel** (with `-DPICKLE_KERNEL`), using a built-in software float32
  layer so it runs under `-mno-sse` with no x87 init. This is the
  in-kernel boot-time selftest shipped as KE-28.
- **On the host** (this repo's default build), using native SSE/AVX/AVX-512
  math, a `mmap` zero-copy loader, OpenMP-threaded quantized matmul, and a
  real Llama BPE tokenizer — a fully independent inference engine that
  closes the gap with llama.cpp/ollama without sharing any code with them.

## Status

**v0.4 alpha.** The Llama-family forward pass is verified end-to-end both
in-kernel (lestraOS boot-time selftest, KE-28) and on the host
(`./pickle selftest`, `./pickle infer`, `./pickle chat`).

Roadmap progress (see [`docs/ROADMAP.md`](docs/ROADMAP.md)):

| Milestone | Status |
|---|---|
| v0.1 — GGUF parser + basic dequant (F32/F16/Q8_0/Q4_0/Q4_1/Q5_0/Q5_1) | ✅ done |
| v0.2 — K-quant family (Q4_K/Q5_K/Q6_K/Q8_K/Q2_K/Q3_K) | ✅ done |
| v0.3 — Llama forward pass (RMSNorm, GQA+RoPE, SwiGLU, greedy/temp sampling) | ✅ done |
| v0.4 — Llama BPE tokenizer with byte-fallback | ✅ done |
| v0.5 — Batched prefill + continuous batching | 🚧 next |
| v1.0 — First stable release | 🚧 planned |

Supported tensor types:

- **Full dequant:** F32, F16, Q8_0, Q4_0, Q4_1, Q5_0, Q5_1
- **Simplified K-quants:** Q4_K, Q5_K, Q6_K, Q8_K, Q2_K, Q3_K

Supported forward pass:

- Llama-family (Llama, Llama 2, Llama 3, Mistral, Qwen2, Phi3-style):
  RMSNorm, GQA attention with RoPE (NeoX), SwiGLU FFN, optional
  logit softcapping, optional tied word embeddings.

## Performance

On a 2-core host with AVX-512 VNNI, running TinyLlama-1.1B-Chat Q4_K_M
(640 MB), the host fast path delivers:

| Metric | Baseline (v0.3) | Optimized (v0.4) | Speedup |
|---|---|---|---|
| Decode (tok/s)         | 7.7  | 11.5 | **+49%** |
| Prefill (tok/s)        | 8.4  | 11.8 | +40% |
| 32-token generation wall | 7.3s | 3.2s | **−56%** |
| 640 MB model startup   | ~5s  | <0.1s | instant |

The five optimizations behind this (all host-only, kernel soft-float path
untouched) landed in commits
[`f4e22b8`](https://github.com/lee-muriethi-kingori/lestramanika/commit/f4e22b8),
[`954be76`](https://github.com/lee-muriethi-kingori/lestramanika/commit/954be76),
[`2aec00a`](https://github.com/lee-muriethi-kingori/lestramanika/commit/2aec00a),
and
[`0e857ed`](https://github.com/lee-muriethi-kingori/lestramanika/commit/0e857ed):

1. **`mmap` zero-copy loader** — the whole GGUF file is `mmap`'d once and
   every tensor data pointer is patched to point straight into the mapping.
   Zero `malloc`, zero `fread`, zero copy. Startup for a 640 MB model drops
   from ~5 s to under 0.1 s. Same idea as llama.cpp's `ggml` mmap backend,
   implemented independently.
2. **Pre-allocated working buffers** — 14 per-token scratch buffers (x, q,
   k, v, scores, gate/up/act/down, …) are allocated once in
   `pickle_fast_state_init` with `posix_memalign` and reused for every
   token. Eliminates 13 `malloc` + 13 `free` per token.
3. **Thread-local x-quant scratch** — the AVX-512 VNNI Q4_K / Q6_K kernels
   reuse per-thread `tls_x_int` / `tls_x_scale` / `tls_sum_x16` buffers
   grown on demand, eliminating 924 allocator calls per token
   (154 matmuls × 6).
4. **AVX-512 vectorized attention** — the score dot-product and V weighted
   sum are unrolled across 4 `zmm` FMAs per position (vs 64 scalar FMAs),
   with OpenMP parallelism across heads.
5. **Software prefetching** in the Q4_K / Q6_K VNNI kernels —
   `_mm_prefetch` of the next two blocks into L2 while the current block
   runs, hiding ~100 ns of DRAM latency per block. The single biggest win
   (+28% on its own).

Quantized matmul has a 3-tier dispatch per type:

```
__AVX512VNNI__  →  hand-rolled VNNI vpdpbusd kernel   (fastest)
__AVX512F__     →  inline 16-wide float FMA kernel
#else           →  scalar / OpenMP-simd fallback
```

`Q4_K` and `Q6_K` both ship VNNI kernels; `Q8_0` / `Q4_0` / `Q5_*` use the
FMA / simd path. The dispatch is chosen at compile time from compiler
defines, so the same binary runs everywhere and just gets faster on newer
hardware.

## Quick start

```sh
make                          # builds ./pickle (CLI) and ./pickle_selftest
./pickle selftest             # runs the embedded selftest → "next token = 6"
./pickle info model.gguf      # prints the architecture + first 12 tensors
./pickle infer model.gguf "hello" 20   # generates 20 tokens from "hello"
./pickle chat  model.gguf              # interactive REPL with BPE tokenizer
./pickle bench  model.gguf "prompt" 32 # decode + prefill tok/s report
./pickle dequant model.gguf token_embd.weight   # first 32 floats of a tensor
```

If you don't have a `model.gguf` handy, regenerate the tiny demo model
that `pickle selftest` uses:

```sh
python3 tools/make_tiny_gguf.py    # writes /tmp/pickle_demo.gguf + src/pickle_demo_gguf.c
./pickle info /tmp/pickle_demo.gguf
./pickle infer /tmp/pickle_demo.gguf "abc" 5
```

For a real model, download any Llama-family Q4_K_M GGUF (TinyLlama-1.1B,
Qwen2-0.5B, Phi-2, etc.) and point `./pickle infer` at it. The host build
uses `mmap` by default, so a 640 MB model loads in well under a second.

### Tuning the host build

```sh
make PICKLE_NO_NATIVE=1    # don't use -march=native (reproducible / cross-arch)
make PICKLE_NO_OMP=1       # disable OpenMP threading
make CFLAGS=-O0            # debug build
```

`-march=native` is on by default so the compiler can emit AVX-512 VNNI
when the CPU supports it. Disable it for reproducible cross-architecture
builds; the FMA / simd fallback still runs.

## Architecture

Pickle is four layers:

```
 ┌────────────────────────────────────────────────────────────────┐
 │  pickle_cli   (src/pickle_cli.c)                                │
 │  — argv parsing, subcommands: selftest | info | infer | chat |  │
 │    bench | dequant                                             │
 ├────────────────────────────────────────────────────────────────┤
 │  pickle_host  (src/pickle_host.c)                  [host only]  │
 │  — POSIX shim: FILE*-based pickle_io_t, malloc allocator,      │
 │    pickle_load_from_file() AND pickle_load_from_file_mmap()    │
 │    (zero-copy mmap), pickle_run_prompt() / pickle_run_chat()   │
 ├────────────────────────────────────────────────────────────────┤
 │  pickle_fast  (src/pickle_fast.c + src/pickle_tokenizer.c +    │
 │                src/pickle_fast.h)                  [host only]  │
 │  — native float math (SSE/AVX/AVX-512 auto-vectorised),        │
 │    Llama BPE tokenizer (SentencePiece '▁' + byte-fallback),    │
 │    quantized Q4_K/Q6_K AVX-512 VNNI matmul, precomputed RoPE,  │
 │    contiguous KV cache, pre-allocated per-token buffers.       │
 │    pickle_forward() dispatches here when !PICKLE_KERNEL.       │
 ├────────────────────────────────────────────────────────────────┤
 │  pickle_core  (src/pickle.c + src/pickle_softfp.c +            │
 │                src/pickle.h)                                    │
 │  — freestanding C, no libc, no syscalls. GGUF parse,           │
 │    dequantize, Llama forward pass (soft-float), greedy/temp    │
 │    sampling. All math goes through sfp_t (uint32_t IEEE-754    │
 │    bit patterns) and the sfp_*() soft-float functions —        │
 │    NEVER C `float` arithmetic, so it links into -mno-sse builds.│
 └────────────────────────────────────────────────────────────────┘
```

**The same `pickle.c` / `pickle_softfp.c` / `pickle.h` source compiles
two ways:**

- **In the lestraOS kernel** — with `-DPICKLE_KERNEL`. Then the
  `#ifdef PICKLE_KERNEL` blocks at the top of each file pull in
  `<lestra/types.h>`, `<lestra/printk.h>`, `<lestra/mm.h>` and the
  kernel bump allocator is used. `pickle_forward()` falls back to the
  soft-float path. This is the in-kernel build at lestraOS commit
  `8d3300c` (KE-28).
- **On the host** (this repo) — with `-UPICKLE_KERNEL` (the default).
  Then those `#ifdef` blocks pull in `<stdint.h>`, `<stdio.h>`,
  `<stdlib.h>`, `<string.h>` instead, `pr_info` is `#define`'d to
  `printf`, `kmalloc` to `malloc`, and so on. The POSIX shim
  (`pickle_host.c`) provides `FILE*`-based and `mmap`-based
  `pickle_io_t` callbacks and a `malloc`-based `pickle_alloc_t`.
  `pickle_forward()` dispatches to the fast path in `pickle_fast.c`.

No glue or `#ifdef _HOST_` is needed inside the core files — the
`PICKLE_KERNEL` toggle handles everything. The fast path is fenced off
behind `#ifndef PICKLE_KERNEL`, so the kernel build never sees native
float math, `mmap`, OpenMP, or AVX-512 intrinsics.

### Repository layout

```
lestramanika/
├── Makefile                       # builds ./pickle and ./pickle_selftest
├── src/                           # the pickle engine source
│   ├── pickle.h                   #   public API (freestanding)
│   ├── pickle.c                   #   GGUF parse + Llama fwd pass (soft-float)
│   ├── pickle_softfp.c            #   IEEE-754 soft float32
│   ├── pickle_fast.h              #   host fast-path API
│   ├── pickle_fast.c              #   AVX-512 VNNI matmul + native forward
│   ├── pickle_tokenizer.c         #   Llama BPE tokenizer (host)
│   ├── pickle_demo_gguf.c         #   embedded tiny GGUF (auto-generated)
│   ├── pickle_host.c              #   POSIX shim (host only)
│   ├── pickle_cli.c               #   CLI frontend (host only)
│   └── pickle_selftest_main.c     #   tiny main() for ./pickle_selftest
├── tools/
│   └── make_tiny_gguf.py          # regenerates the embedded demo model
└── docs/
    ├── ARCHITECTURE.md
    └── ROADMAP.md
```

> **Why does source live under `src/` rather than `pickle/`?** Because
> the CLI binary is `./pickle` at the repo root, and a POSIX filesystem
> cannot have both a `pickle/` directory and a `pickle` file at the same
> level (a directory entry name is unique). Putting source under `src/`
> lets the verification commands `./pickle selftest`, `./pickle info …`,
> `./pickle infer …`, `./pickle dequant …` work exactly as written.

## Why

Because the lestraOS kernel disables SSE (`-mno-sse`) and has no x87
init, so any `float` arithmetic would `#NM`/`#UD`. Pickle ships its own
software float32 layer (`pickle_softfp.c`, ~370 lines of integer-only
IEEE-754 binary32 add/sub/mul/div/exp/tanh/sigmoid/silu/gelu/sqrt/rsqrt/
sin/cos) so the **same core source** runs anywhere — kernel, embedded,
anywhere. On the host, where SSE/AVX are available, the fast path
(`pickle_fast.c`) layers on top of that same core to deliver
production-grade throughput with hand-tuned AVX-512 VNNI kernels.

The same engine that boots inside the lestraOS kernel (verifying the
GGUF parser and Llama forward pass at boot time, with no filesystem
access, using an embedded 4 KB demo model) is the one you build with
`make` here — and the host build adds the mmap loader, BPE tokenizer,
and VNNI matmul on top.

## CLI reference

```
pickle selftest                              Run the embedded selftest
pickle info <model.gguf>                     Print model architecture + tensors
pickle infer <model.gguf> "<prompt>" [N]     Generate N tokens (default 20)
pickle chat  <model.gguf>                    Interactive REPL (BPE tokenizer)
pickle bench <model.gguf> "<prompt>" [N]     Print decode + prefill tok/s
pickle dequant <model.gguf> <tensor>         Print first 32 floats of <tensor>
```

Exit codes: `0` on success, `1` on error.

The `infer` / `chat` / `bench` subcommands use the real **Llama BPE
tokenizer** (read from `tokenizer.ggml.*` GGUF metadata, with
SentencePiece `▁` handling and byte-fallback). The `selftest`
subcommand uses a character-level tokenizer on the embedded demo model
— sufficient to exercise the forward pass end to end without shipping a
full vocab.

## Building from a fresh checkout

Requirements: a C compiler (`cc`/`gcc`/`clang`), `make`, and Python 3
(only if you want to regenerate the demo GGUF). OpenMP is optional but
on by default; AVX-512 is auto-detected from `-march=native`.

```sh
make clean && make
make test    # builds + runs ./pickle_selftest and ./pickle selftest
make bench   # builds + runs a quick benchmark on the demo model
```

`CFLAGS`, `LDFLAGS`, and `CC` are overridable on the make command line
or via the environment.

## Relationship to lestraOS

lestraOS vendors the **kernel-compatible** half of this repo (the core
`pickle.c`, `pickle_softfp.c`, `pickle.h`, and `pickle_demo_gguf.c`)
into `kernel/ai/` and `kernel/include/lestra/`. The fast path
(`pickle_fast.c`, `pickle_tokenizer.c`, `pickle_host.c`, `pickle_cli.c`)
is host-only and lives exclusively in this repo.

lestraOS tracks this repo as a git submodule at
`third_party/lestramanika` and ships a `scripts/sync_lestramanika.sh`
that copies the kernel-compatible sources into the kernel tree. See
lestraOS's [`docs/LESTRAMANIKA.md`](https://github.com/lee-muriethi-kingori/LestraOS/blob/main/docs/LESTRAMANIKA.md)
for the integration guide.

## License

MIT, Copyright (c) 2026 Lee Muriihi Kingori. See [LICENSE](LICENSE).

## Repo

- **This repo:** https://github.com/lee-muriethi-kingori/lestramanika

## Related

- **lestraOS:** https://github.com/lee-muriethi-kingori/LestraOS —
  the OS where pickle runs in-kernel. The in-kernel half of this work
  shipped as KE-28 (commit `8d3300c`): a boot-time selftest that parses
  the embedded demo GGUF and runs one Llama forward pass, printing
  `pickle: selftest OK, next token = 6` to the kernel console.
