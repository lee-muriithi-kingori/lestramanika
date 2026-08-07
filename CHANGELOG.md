# Changelog

All notable changes to **pickle** are documented in this file. The format is
based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/), and this
project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

A release is cut as an annotated git tag `vMAJOR.MINOR.PATCH` with a matching
GitHub Release; the version string reported by `pickle --version` is the single
source of truth and is generated from `src/version.h`.

## [1.0.0] — 2026-08-07

First stable release of the host inference engine. The single-sequence Llama
forward pass, the BPE tokenizer, and the full CLI are now considered stable:
their command-line interface and the `pickle_core` / `pickle_host` C APIs are
frozen for the 1.x line. Batched prefill and continuous batching remain future
work (tracked under v0.5 / 2.0) and are not part of the 1.0 stability surface.

### Added
- `pickle version`, `pickle --version`, and `pickle -V` now print the version
  string (`pickle 1.0.0`), sourced from the new `src/version.h`.
- `CHANGELOG.md` tracking every release from v0.1.0 onward.
- `CONTRIBUTING.md` with build, test, and contribution guidance.
- `SECURITY.md` with a responsible-disclosure policy.
- GitHub Actions continuous-integration workflow (`.github/workflows/ci.yml`)
  that builds `pickle` + `pickle_selftest` and runs `make test` on Linux and
  macOS for every push and pull request.

### Changed
- README status bumped from "v0.4 alpha" to "v1.0.0 stable"; the roadmap table
  now marks v1.0 as **Done**.

### Stability guarantees (1.x line)
- CLI subcommands (`selftest`, `info`, `infer`, `chat`, `bench`, `tokens`,
  `dequant`, `version`) and their flags (`--temp`, `--top-p`, `--seed`,
  `--max`, `--no-bos`) will not be removed or renamed. New subcommands and
  flags may be added (minor bumps); behaviour changes for existing flags on
  existing valid input will not happen without a major bump.
- The `pickle_core` API declared in `src/pickle.h` and the `pickle_host` /
  `pickle_fast` / tokenizer APIs are frozen for the shapes and names they
  expose today.

### Verification
- `make` builds `pickle` and `pickle_selftest` with zero warnings under
  `-Wall -Wextra -O3 -march=native -fopenmp`.
- `./pickle selftest` and `./pickle_selftest` both report
  `pickle: selftest OK, next token = 5` and exit 0.
- `./pickle --version` prints `pickle 1.0.0`.

---

## [0.4.0] — 2026-07

### Added
- mmap zero-copy loader (`pickle_load_from_file_mmap`) with pre-allocated
  activation buffers, eliminating per-step `malloc`/`free` churn.
- AVX-512 attention kernel with software prefetching.
- Host BPE tokenizer (`src/pickle_tokenizer.c`) read directly from the
  `tokenizer.ggml.*` GGUF metadata: SentencePiece `▁` (U+2581) handling,
  byte-fallback, BOS prepend, and both encode/decode directions.
- `pickle tokens <model> encode|decode …` subcommand.
- Streaming `pickle chat` REPL with `:quit` / `:reset`.
- `make bench` target and the `pickle bench` subcommand reporting prefill
  tok/s and decode tok/s.

### Changed
- Fast-path decode throughput on TinyLlama-1.1B Q4_K_M improved ~49% over
  v0.3 on a 2-core AVX-512 VNNI host.

### Fixed
- mmap + quantized-embedding bug that produced all-zero logits and
  incoherent output (7ab120a).

---

## [0.3.0] — 2026-06

### Added
- Llama-family forward pass: RMSNorm (per-layer eps from metadata), grouped-
  query attention with RoPE (NeoX convention), SwiGLU MLP, optional logit
  softcapping (Gemma-style), optional tied word embeddings.
- Greedy and temperature / nucleus (top-p) sampling.
- KV cache.
- `pickle infer` and `pickle chat` end-to-end on real GGUF models.

---

## [0.2.0] — 2026-05

### Added
- K-quant dequantizers: `Q4_K`, `Q5_K`, `Q6_K`, `Q8_K`, `Q2_K`, `Q3_K`.
- Super-block layout, min/max/scale decoding.

---

## [0.1.0] — 2026-04

### Added
- GGUF v3 container reader: magic, version, header, key/value table, tensor
  info table, with little-endian-on-disk handling for both LE and BE hosts.
- mmap-free byte-buffer API so the same parser runs on the host and in a
  kernel (`-DPICKLE_KERNEL`).
- Dequantizers for `F32`, `F16`, `Q8_0`, `Q4_0`, `Q4_1`, `Q5_0`, `Q5_1`.
- `pickle info` and `pickle dequant` subcommands.
- Embedded demo GGUF (`src/pickle_demo_gguf.c`) + `pickle selftest` /
  `pickle_selftest` covering the full forward pass end to end.

[1.0.0]: https://github.com/lee-muriithi-kingori/picklestramk/releases/tag/v1.0.0
[0.4.0]: https://github.com/lee-muriithi-kingori/picklestramk/releases/tag/v0.4.0
[0.3.0]: https://github.com/lee-muriithi-kingori/picklestramk/releases/tag/v0.3.0
[0.2.0]: https://github.com/lee-muriithi-kingori/picklestramk/releases/tag/v0.2.0
[0.1.0]: https://github.com/lee-muriithi-kingori/picklestramk/releases/tag/v0.1.0
