# Roadmap

This is the version plan for pickle. Each minor version is a self-contained,
testable milestone. Nothing ships until the milestone below it is done.

## v0.1 — Parser and basic dequantisation

- GGUF container reader: magic, version, header, kv-table, tensor-info table.
- Endianness handling (GGUF is little-endian on disk; pickle reads on both
  LE and BE hosts).
- mmap-free byte-buffer API so the same parser works on the host and in a
  kernel.
- Dequantisers for the four "easy" formats: **F32, F16, Q8_0, Q4_0**.
- Round-trip tests: parse a real GGUF file, dequant a tensor, compare against
  a reference.

**Done when:** `pickle inspect model.gguf` lists every tensor with its name,
shape, and dtype, and `pickle dequant model.gguf tensor.npq` dumps a tensor's
F32 values to stdout for any of the four formats above.

## v0.2 — K-quant family

- Dequantisers for **Q4_K, Q5_K, Q6_K** (and Q4_K_S / Q5_K_S variants).
- Super-block layout handling, min/max/scale decoding.
- Bit-exact equivalence tests against reference values.

**Done when:** every K-quant tensor in a real Llama-style model dequantises to
values matching the reference within 1 ULP on F32.

## v0.3 — Llama forward pass

- Tensor view + slice helpers (no full BLAS — small, readable matmul).
- Token embedding lookup.
- RMSNorm.
- RoPE (rotary positional embedding), both GPT-NeoX and GPT-J conventions.
- Grouped-query attention with a KV cache.
- SwiGLU MLP.
- Output projection + argmax.

**Done when:** given a prompt and a Llama-architecture GGUF model in a supported
quant, pickle produces the same next-token distribution as a reference
implementation for the first 32 tokens, greedy.

## v0.4 — BPE tokenizer

- Llama / SentencePiece BPE tokenizer reader from the GGUF metadata.
- Encode + decode, including special tokens and byte-fallback.
- Pre-tokenizer normalisation rules (the Llama-specific ones actually used in
  practice).

**Done when:** a prompt can be tokenised, run through the model from v0.3, and
detokenised back to text, end to end, on the host.

## v0.5 — Batching

- Batched prefill: process a prompt's tokens in one forward pass.
- Continuous batching of multiple sequences (single-threaded, fair-schedule).
- Shared KV cache management with per-sequence slots.

**Done when:** throughput on a single sequence is at least 1.5× the
v0.3 one-token-at-a-time loop, and two interleaved sequences produce the same
outputs as two separate runs.

## v1.0 — First release (DONE — shipped 2026-08-07 as tag `v1.0.0`)

- `pickle_cli` REPL with streaming generation.
- Deterministic mode (fixed seed `--seed`, fixed temperature schedule) for
  reproducible tests.
- A reference model + prompt set: the embedded 4 KiB demo GGUF
  (`src/pickle_demo_gguf.c`) + `pickle selftest` / `pickle_selftest`, which
  exercise the full forward pass end to end without shipping a large vocab.
- API stability guarantees for `pickle_core` (`src/pickle.h`) and
  `pickle_host` / `pickle_fast` / the BPE tokenizer — frozen for the 1.x line.
- `pickle version` / `--version` / `-V`, sourced from `src/version.h`.
- `CHANGELOG.md`, `CONTRIBUTING.md`, `SECURITY.md`, and a GitHub Actions CI
  workflow (`.github/workflows/ci.yml`) that builds + selftests on every push
  and pull request, on Linux and macOS.
- This README updated with build instructions, supported models, and
  benchmarks.

**Done when:** a third party can `git clone`, build with one command on a stock
Linux/macOS box, and run a supported model from a single CLI invocation.

**Scope note:** the single-sequence path, the BPE tokenizer, and the full CLI
are the 1.0 stability surface. Batched prefill and continuous batching (v0.5)
and multi-turn chat context are explicitly **not** part of 1.0 and remain
future work (a 2.0 line).
