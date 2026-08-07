# Contributing to pickle

pickle is a from-scratch GGUF loader and transformer inference engine in
freestanding C. It is deliberately small and readable; the bar for a change
is that it keeps the source readable, the build warning-clean, and the
selftest green. This document explains how to do that.

## 1. Build and test locally

You need a C11 compiler (`cc`/`gcc`/`clang`) and `make`. OpenMP is optional
but recommended for the threaded matmul.

```sh
make            # builds ./pickle and ./pickle_selftest
make test       # builds, then runs ./pickle_selftest and ./pickle selftest
make clean
```

A change is not ready until **all** of these pass with zero warnings under
`-Wall -Wextra -O3`:

```sh
make clean && make -j
./pickle_selftest                       # expect: selftest rc=0 token=5
./pickle selftest                       # expect: pickle: selftest OK, next token = 5
./pickle --version                      # expect: pickle 1.0.0
make clean && make -j PICKLE_NO_NATIVE=1 PICKLE_NO_OMP=1   # portable build
```

The portable build (`PICKLE_NO_NATIVE=1 PICKLE_NO_OMP=1`) is how the source
is consumed in the lestraOS kernel and on constrained hosts — it must still
compile and selftest must still pass.

## 2. The source stays dual-build

`src/pickle.c`, `src/pickle_softfp.c`, and `src/pickle.h` are compiled two
ways from the same bytes:

- **Host** (`-UPICKLE_KERNEL`, this repo) — uses `<stdio.h>`/`<stdlib.h>`,
  native SSE/AVX/AVX-512, `mmap`, OpenMP.
- **Kernel** (`-DPICKLE_KERNEL`, in lestraOS) — uses lestra kernel headers
  and a software float32 layer under `-mno-sse`.

Do **not** introduce a `#ifdef` that forks behaviour between the two builds
beyond what `pickle.h` already gates. If you must, add the gate to `pickle.h`
next to the existing `PICKLE_KERNEL` block and keep the host and kernel paths
visibly parallel.

## 3. What a good change looks like

- **Correctness first.** pickle is an inference engine; a 1% speedup that
  changes logits is a regression. If you touch the forward pass, show that
  `./pickle selftest` still prints `next token = 5` and, where relevant,
  that a real model's greedy output is byte-identical before and after.
- **No new dependencies.** The whole point is "no llama.cpp, no ggml, no
  ollama". If you think you need one, open an issue first.
- **Warnings are failures.** Do not silence a warning with a cast unless you
  understand why it appeared. Prefer fixing the root cause.
- **Keep it readable.** This codebase is meant to be readable by someone
  learning how a transformer works. A 40-line clear function beats a 10-line
  clever one.

## 4. Commit messages

Use the conventional style already in the history:

```
<area>: <imperative summary of the change>

<optional body explaining why, not what>
```

`<area>` is one of `pickle`, `pickle_cli`, `pickle_fast`, `tokenizer`,
`docs`, `build`, `ci`, `test`. Example:

```
pickle_fast: hoist the Q6_K scale lookup out of the inner loop

The per-superblock scale was being re-read on every 32-element block;
reading it once per superblock is 4% faster on Zen 4 with no logit change.
```

## 5. Cutting a release (maintainers)

1. Confirm `make test` is green on `main` and CI is green.
2. Bump `src/version.h` (`MAJOR`/`MINOR`/`PATCH`) per the policy in that file.
3. Add a `## [x.y.z] — <date>` section to `CHANGELOG.md` and bump the
   `[Unreleased]`/status lines in `README.md`.
4. Commit, push, then tag and push an annotated tag:
   ```sh
   git tag -a v1.0.0 -m "pickle 1.0.0 — first stable release"
   git push origin v1.0.0
   ```
5. Create the GitHub Release from the tag with the changelog section as the
   body. The tag triggers the `on: tags: ["v*"]` CI run.

## 6. Reporting issues

For bugs, include: OS, CPU (`gcc -march=native -Q --help=target | grep march`),
compiler version, the exact command you ran, and the full output of
`./pickle selftest` and (if relevant) `./pickle info <model.gguf>`.

For security issues, see `SECURITY.md` — do **not** open a public issue.

Thank you for helping make pickle better.
