# Security policy

pickle is a from-scratch GGUF model loader and transformer inference engine.
This document explains what is and is not in scope for security reports and
how to report a vulnerability responsibly.

## Supported versions

Only the latest tagged release on `main` receives security fixes. The
version is the one reported by `pickle --version` (e.g. `pickle 1.0.0`),
matching the most recent `v*` git tag.

| Version | Supported |
|---------|-----------|
| 1.0.x   | Yes       |
| < 1.0   | No (pre-release alpha milestones) |

## Reporting a vulnerability

**Do not open a public GitHub issue for a security vulnerability.**

Instead, report it privately using GitHub's built-in advisory feature:

1. Go to
   https://github.com/lee-muriithi-kingori/picklestramk/security/advisories/new
2. Click **"Report a vulnerability"**.
3. Fill in a title, a description of the issue and its impact, and a
   minimal reproduction (a `.gguf` file or a command line is ideal).

You should receive an acknowledgement within 72 hours. If the report is
accepted, we will coordinate a fix and a disclosure timeline with you, and
credit you in the advisory and the `CHANGELOG.md` entry unless you prefer
to remain anonymous.

If you cannot use GitHub private advisories for some reason, you may
instead email the maintainer at `lee@lestramk.org` with the subject line
`[pickle security] <short summary>`. Please do **not** attach large model
files to the email; host them somewhere and share a link.

## Scope

### In scope

- **Malformed-GGUF parsing.** pickle parses untrusted `.gguf` files.
  A crash, out-of-bounds read/write, infinite loop, or integer overflow in
  the parser (`src/pickle.c`, the `pickle_load_from_file*` family in
  `src/pickle_host.c`) triggered by a crafted file is in scope.
- **Dequantizer memory safety.** Out-of-bounds access in any dequantizer
  (Q4_0/Q4_1/Q5_0/Q5_1/Q8_0, the K-quants, F16) when fed a tensor whose
  on-disk byte length does not match its declared shape/type.
- **Tokenizer bugs with security impact.** A crafted `tokenizer.ggml.*`
  metadata block that causes an out-of-bounds access in
  `src/pickle_tokenizer.c`.
- **The CLI frontend** (`src/pickle_cli.c`) leaking memory or crashing on
  hostile command-line input.

### Out of scope

- **Model output quality.** pickle is a deterministic function of its inputs.
  "The model said something offensive / wrong / hallucinated" is not a
  pickle bug — report it to the model's publisher.
- **Performance.** A slow parser path is not a security issue; open a normal
  issue or a PR.
- **Denial of service via a legitimately huge model.** Loading a 50 GB model
  on a machine with 4 GB of RAM will fail; that is expected. A *crafted
  small* file that crashes the parser, however, is in scope.
- **The lestraOS in-kernel build.** The `-DPICKLE_KERNEL` build is tracked in
  the [lestraOS](https://github.com/lee-muriithi-kingori/LestraOS) repository;
  report kernel-side issues there.
- **Issues in dependencies we do not have.** pickle depends only on the C
  standard library, libm, and (optionally) OpenMP. There is no llama.cpp,
  no ggml, no protobuf, no JSON parser to attack.

## Hardening notes for users

- pickle does not execute model code. A `.gguf` file is data; it is parsed
  and its weights are fed through a fixed forward pass. There is no scripting
  or plugin layer.
- pickle does not make network connections. The only file it opens is the
  one you pass on the command line.
- `mmap` is used read-only; pickle never writes to a model file.

## Disclosure

We follow coordinated disclosure: once a fix is available in a tagged
release, we publish a GitHub Security Advisory with full details and credit
the reporter. We aim to ship a fix within 30 days of confirmation for
high-severity issues, faster for anything that is being actively exploited.
