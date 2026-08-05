# pickle — a from-scratch GGUF model loader and inference engine

> No llama.cpp. No ollama. No ggml. Pure C. Runs in the lestraOS kernel and on the host.

`pickle` is a tiny, self-contained GGUF (v3) parser + transformer
forward-pass engine written in freestanding C. It does **not** depend on
llama.cpp, ggml, ollama, or any other inference framework. The math is
done with a built-in software float32 layer (`pickle_softfp.c`) so it
runs even when the host CPU's FPU/SSE is unavailable — for example,
inside the [lestraOS](https://github.com/lee-muriithi-kingori/LestraOS)
kernel, which is built with `-mno-sse` and has no x87 init.

## Status

**alpha.** The Llama-family forward pass is verified end-to-end both
in-kernel (lestraOS boot-time selftest, commit
[`8d3300c`](https://github.com/lee-muriithi-kingori/LestraOS/commit/8d3300c)
/ KE-28) and on the host (this repo's `./pickle selftest`).

Supported tensor types:

- **Full dequant:** F32, F16, Q8_0, Q4_0, Q4_1, Q5_0, Q5_1
- **Simplified K-quants:** Q4_K, Q5_K, Q6_K, Q8_K, Q2_K, Q3_K

Supported forward pass:

- Llama-family (Llama, Llama 2, Llama 3, Mistral, Qwen2, Phi3-style):
  RMSNorm, GQA attention with RoPE (NeoX), SwiGLU FFN, optional
  logit softcapping, optional tied word embeddings.

## Quick start

```sh
make                          # builds ./pickle (CLI) and ./pickle_selftest
./pickle selftest             # runs the embedded selftest → "next token = 6"
./pickle info model.gguf      # prints the architecture + first 12 tensors
./pickle infer model.gguf "hello" 20   # generates 20 tokens from "hello"
./pickle dequant model.gguf token_embd.weight   # first 32 floats of a tensor
```

If you don't have a `model.gguf` handy, regenerate the tiny demo model
that `pickle selftest` uses:

```sh
python3 tools/make_tiny_gguf.py    # writes /tmp/pickle_demo.gguf + src/pickle_demo_gguf.c
./pickle info /tmp/pickle_demo.gguf
./pickle infer /tmp/pickle_demo.gguf "abc" 5
```

## Architecture

Pickle is three layers:

```
 ┌──────────────────────────────────────────────────────────────┐
 │  pickle_cli   (src/pickle_cli.c)                              │
 │  — argv parsing, subcommands: selftest | info | infer | dequant │
 ├──────────────────────────────────────────────────────────────┤
 │  pickle_host  (src/pickle_host.c)              [host only]    │
 │  — POSIX shim: FILE*-based pickle_io_t, malloc allocator,    │
 │    pickle_load_from_file(), pickle_run_prompt() (char-level  │
 │    tokenizer + greedy generation loop)                       │
 ├──────────────────────────────────────────────────────────────┤
 │  pickle_core  (src/pickle.c + src/pickle_softfp.c +           │
 │                src/pickle.h)                                  │
 │  — freestanding C, no libc, no syscalls. GGUF parse,         │
 │    dequantize, Llama forward pass, greedy sampling. All math │
 │    goes through sfp_t (uint32_t IEEE-754 bit patterns) and   │
 │    the sfp_*() soft-float functions — NEVER C `float`        │
 │    arithmetic, so it links into -mno-sse builds.             │
 └──────────────────────────────────────────────────────────────┘
```

**The same `pickle.c` / `pickle_softfp.c` / `pickle.h` source compiles
two ways:**

- **In the lestraOS kernel** — with `-DPICKLE_KERNEL`. Then the
  `#ifdef PICKLE_KERNEL` blocks at the top of each file pull in
  `<lestra/types.h>`, `<lestra/printk.h>`, `<lestra/mm.h>` and the
  kernel bump allocator is used. This is the in-kernel build at
  lestraOS commit `8d3300c` (KE-28).
- **On the host** (this repo) — with `-UPICKLE_KERNEL` (the default).
  Then those `#ifdef` blocks pull in `<stdint.h>`, `<stdio.h>`,
  `<stdlib.h>`, `<string.h>` instead, `pr_info` is `#define`'d to
  `printf`, `kmalloc` to `malloc`, and so on. The POSIX shim
  (`pickle_host.c`) provides `FILE*`-based `pickle_io_t` callbacks
  and a `malloc`-based `pickle_alloc_t`.

No glue or `#ifdef _HOST_` is needed inside the core files — the
`PICKLE_KERNEL` toggle handles everything.

### Repository layout

```
lestramanika/
├── Makefile                       # builds ./pickle and ./pickle_selftest
├── src/                           # the pickle engine source
│   ├── pickle.h                   #   public API (freestanding)
│   ├── pickle.c                   #   GGUF parse + Llama fwd pass
│   ├── pickle_softfp.c            #   IEEE-754 soft float32
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
sin/cos) so it runs anywhere — kernel, embedded, anywhere.

The same engine that boots inside the lestraOS kernel (verifying the
GGUF parser and Llama forward pass at boot time, with no filesystem
access, using an embedded 4 KB demo model) is the one you build with
`make` here.

## CLI reference

```
pickle selftest                         Run the embedded selftest
pickle info <model.gguf>                Print model architecture + tensors
pickle infer <model.gguf> "<prompt>" [N] Generate N tokens (default 20)
pickle dequant <model.gguf> <tensor>    Print first 32 floats of <tensor>
```

Exit codes: `0` on success, `1` on error.

The `infer` subcommand uses a **character-level tokenizer** for the demo
model (each prompt byte becomes a token id, mod `vocab_size`). This is
sufficient to exercise the forward pass end-to-end. A real BPE tokenizer
is a v0.4 roadmap item.

## Building from a fresh checkout

Requirements: a C compiler (`cc`/`gcc`/`clang`), `make`, and Python 3
(only if you want to regenerate the demo GGUF).

```sh
make clean && make
make test    # builds + runs ./pickle_selftest and ./pickle selftest
```

`CFLAGS`, `LDFLAGS`, and `CC` are overridable on the make command line
or via the environment.

## License

MIT, Copyright (c) 2026 Lee Muriihi Kingori. See [LICENSE](LICENSE).

## Repo

- **This repo:** https://github.com/lee-muriithi-kingori/lestramanika

## Related

- **lestraOS:** https://github.com/lee-muriithi-kingori/LestraOS —
  the OS where pickle runs in-kernel. The in-kernel half of this work
  shipped as KE-28 (commit `8d3300c`): a boot-time selftest that parses
  the embedded demo GGUF and runs one Llama forward pass, printing
  `pickle: selftest OK, next token = 6` to the kernel console.
