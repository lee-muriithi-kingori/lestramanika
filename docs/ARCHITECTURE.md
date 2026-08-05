# Architecture

pickle is built in three layers. Each layer is a strict superset of the one
below it: the higher layer may call the lower layer, never the reverse, and
never sideways. The boundary lines are enforced by link-time discipline (each
layer is its own translation unit / archive) and by a small set of header
contracts.

```
┌─────────────────────────────────────────────────────┐
│  pickle_cli                                         │
│  argv parsing · REPL · I/O · generation loop         │  host-facing
├─────────────────────────────────────────────────────┤
│  pickle_host                                        │
│  open/read/mmap · malloc/free · clock · stderr       │  host-facing
├─────────────────────────────────────────────────────┤
│  pickle_core                                        │
│  GGUF parse · dequant · tensor math · forward pass   │  freestanding
└─────────────────────────────────────────────────────┘
```

## `pickle_core` — freestanding

`pickle_core` is the entire engine: the GGUF parser, every dequantiser, the
tensor math, and the forward pass. It has these properties:

- **No libc.** It does not include `<stdio.h>`, `<stdlib.h>`, `<string.h>`,
  `<math.h>`, or any other libc header. It provides its own `memcpy`/`memset`
  and its own soft-float math where needed.
- **No syscalls.** It does not know what a file, a socket, or a thread is.
- **No allocations it does not own.** All memory comes from an `arena` pointer
  the caller threads in. The core never calls `malloc`.
- **No SIMD by default.** The numerical kernels use a portable, scalar,
  soft-float path. A vectorised fast path is selected at compile time via a
  `PICKLE_HAVE_SSE2` / `PICKLE_HAVE_AVX2` switch; the host shim sets this when
  it has queried the CPU. The scalar path is always compiled in and is the only
  path used when SIMD is unavailable — for example, inside the Linux kernel,
  where `kernel_fpu_begin()` is expensive and SIMD is often left off.

The public surface of `pickle_core` is small:

- `pickle_arena` — a bump allocator over a caller-supplied byte buffer.
- `pickle_gguf` — a parsed GGUF file: header, kv-table, tensor-info table, and
  a view over the raw tensor data.
- `pickle_tensor` — a dequantised tensor view (always F32 in memory).
- `pickle_dequant_*` — one function per on-disk dtype (F32, F16, Q8_0, Q4_0,
  Q4_K, Q5_K, Q6_K, …).
- `pickle_forward` — the Llama forward pass operating on `pickle_tensor`s and
  a `pickle_kv_cache`.

Because `pickle_core` is freestanding, the **same object file** can be linked
into:

- the host CLI (via `pickle_host`),
- a unikernel image,
- a Linux kernel module (with `pickle_khost` providing the arena and the
  `read` callback),
- a WASM module (with a tiny WASI shim).

## `pickle_host` — POSIX shim

`pickle_host` is the only layer that knows about an operating system. Its job
is to give `pickle_core` what it needs to run on a normal POSIX host:

- `pickle_host_read_file(path, arena)` — `open` + `fstat` + `read` (or `mmap`)
  into the arena; return a byte buffer.
- `pickle_host_arena_reserve(size)` — back an arena with `malloc`-ed memory.
- `pickle_host_clock_us()` — a monotonic microsecond clock for timing.
- `pickle_host_log(...)` — `fprintf(stderr, ...)` for diagnostics.
- CPU feature detection (`pickle_host_cpu_features()`) so the build can pick
  the SSE2/AVX2 fast path at runtime instead of compile time.

`pickle_host` does **not** contain any GGUF or model code. If you swap it out
for `pickle_khost` (kernel) or a WASI shim, `pickle_core` runs unchanged.

## `pickle_cli` — frontend

`pickle_cli` is the user-facing binary. It:

- parses `argv` (`pickle inspect`, `pickle dequant`, `pickle run`, …),
- calls `pickle_host` to read the model file into an arena,
- hands the buffer to `pickle_core` to parse,
- runs the forward pass via `pickle_core`,
- streams tokens to stdout via `pickle_host`.

`pickle_cli` is the only layer that is allowed to be platform-specific in its
*user-facing* behaviour (signal handling, terminal raw mode for the REPL, etc.).
Everything below it is portable.

## Why this layering?

1. **Auditability.** The engine is one C tree you can read top to bottom. There
   is no C++, no templates, no GPU runtime, no plugin system. A reader who
   understands GGUF can understand pickle.
2. **Kernel-safety.** Because `pickle_core` is freestanding and soft-float, it
   can be linked into a Linux kernel module and used to serve a small language
   model from ring 0. That is the project's long-term target. The layering
   makes "pickle in a kernel" a build-time choice, not a rewrite.
3. **No hidden dependencies.** `pickle_core` cannot accidentally call `malloc`
   or `printf` because they are not in its link line. The build will fail
   loudly the moment someone tries.

## Build layout (planned)

```
pickle/
├── core/        # pickle_core — freestanding
│   ├── gguf.c
│   ├── dequant_f32.c
│   ├── dequant_f16.c
│   ├── dequant_q8_0.c
│   ├── dequant_q4_0.c
│   ├── dequant_q4_k.c
│   ├── dequant_q5_k.c
│   ├── dequant_q6_k.c
│   ├── tensor.c
│   ├── forward.c
│   └── arena.c
├── host/        # pickle_host — POSIX shim
│   └── host_posix.c
├── cli/         # pickle_cli — frontend
│   └── main.c
└── include/
    └── pickle/
        ├── core.h
        ├── host.h
        └── cli.h
```

The `pickle/` directory is intentionally empty in the initial skeleton; it is
populated by the first development sprint targeting `v0.1` (see
[`ROADMAP.md`](./ROADMAP.md)).
