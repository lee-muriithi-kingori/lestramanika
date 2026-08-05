# pickle

**pickle** is a from-scratch GGUF model loader and inference engine.

- **No llama.cpp.** Zero lines of llama.cpp code. No vendored llama.cpp.
- **No ollama.** Not a wrapper, not a fork, not a client. No ollama dependency.
- **Pure C.** One language, one compiler, one toolchain.
- **Freestanding core.** The model loader and inference engine (`pickle_core`)
  depend on **no libc** and **no operating system**. It compiles to a freestanding
  object you can link into anything — a userspace binary, a unikernel, or a
  kernel module.
- **Soft-float by default.** The numerical kernels use a portable software
  floating-point path, so they run on targets where SSE/AVX are unavailable or
  disabled — including ring-0 kernel code where SIMD is intentionally turned off
  on Linux. A vectorised fast path is selected at build time when the host
  advertises it.
- **Host shim.** `pickle_host` is a thin POSIX shim that gives the freestanding
  core `malloc`, `read`, `write`, and a clock. That is the entire host surface.

> pickle is **work in progress.** It does not run a model end-to-end yet.
> See [`docs/ROADMAP.md`](docs/ROADMAP.md) for the version plan and
> [`docs/ARCHITECTURE.md`](docs/ARCHITECTURE.md) for the layering.

## What is GGUF?

[GGUF](https://github.com/ggerganov/ggml/blob/master/docs/gguf.md) (GPT-Generated
Unified Format) is the binary model container used by the modern GGML ecosystem.
A GGUF file is a single key/value metadata header followed by a flat tensor
table. Tensors are stored in a family of compact quantised formats (F32, F16,
Q8_0, Q4_0, Q4_K, Q5_K, Q6_K, …) that let multi-billion-parameter models fit in
a few gigabytes of RAM.

pickle reads that container directly — parsing the header, decoding the tensor
table, and dequantising tensors on demand — without any third-party GGUF code.

## Why?

Two reasons:

1. **To prove it can be done cleanly.** GGUF and the surrounding quantisation
   formats are small, well-specified, and self-contained. There is no reason
   an inference engine needs to drag in megabytes of C++ and a GPU abstraction
   layer. pickle aims to be a readable, single-tree implementation that a single
   person can hold in their head.
2. **To run inside a kernel.** Because `pickle_core` is freestanding and
   soft-float, it can be linked into a kernel module and used to serve a small
   language model from ring 0. That is the long-term target.

## Layering

```
┌─────────────────────────────────────────────┐
│  pickle_cli   — argument parsing, REPL, I/O  │  host-facing
├─────────────────────────────────────────────┤
│  pickle_host  — POSIX shim: malloc/read/write │  host-facing
├─────────────────────────────────────────────┤
│  pickle_core  — GGUF parse + dequant + math   │  freestanding
└─────────────────────────────────────────────┘
```

- `pickle_core` is freestanding C. It calls no libc function. It does not know
  what a file is. It accepts byte buffers and arena allocators.
- `pickle_host` is the POSIX shim: it opens files, calls `mmap`/`read`,
  provides `malloc`/`free`, and exposes a monotonic clock. It is the only layer
  that knows about the operating system.
- `pickle_cli` is the frontend: it parses argv, wires `pickle_host` to
  `pickle_core`, and drives an interactive REPL or a one-shot generation.

See [`docs/ARCHITECTURE.md`](docs/ARCHITECTURE.md) for the full breakdown.

## Status

**WIP — pre-release.** The current tree contains the skeleton only. The first
working milestone (`v0.1`: GGUF parser + F32/F16/Q8_0/Q4_0 dequant) is the
target of the initial development sprint. See
[`docs/ROADMAP.md`](docs/ROADMAP.md).

## License

MIT — see [`LICENSE`](LICENSE).

## Author

**Lee Muriihi Kingori** — lee-muriethi-kingori.
