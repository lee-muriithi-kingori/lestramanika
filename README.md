<h1>pickle</h1>

<blockquote>
<p><strong>A from-scratch GGUF model loader and transformer inference engine.</strong><br>
No llama.cpp. No ollama. No ggml. Pure C. Runs in the lestraOS kernel and on the host.</p>
</blockquote>

<p><code>pickle</code> is a tiny, self-contained GGUF (v3) parser plus a
transformer forward-pass engine written in freestanding C. It does
<em>not</em> depend on llama.cpp, ggml, ollama, or any other inference
framework &mdash; every line of the parser, dequantizer, tokenizer, and
forward pass lives in this repository. The same source compiles two ways:</p>

<ul>
<li><p><strong>In the <a href="https://github.com/lee-muriethi-kingori/LestraOS">lestraOS</a> kernel</strong>
(with <code>-DPICKLE_KERNEL</code>), using a built-in software float32
layer so it runs under <code>-mno-sse</code> with no x87 init. This is the
in-kernel boot-time selftest shipped as KE-28.</p></li>
<li><p><strong>On the host</strong> (this repository&rsquo;s default build), using
native SSE/AVX/AVX-512 math, a <code>mmap</code> zero-copy loader,
OpenMP-threaded quantized matmul, and a real Llama BPE tokenizer &mdash; a
fully independent inference engine that closes the gap with
llama.cpp/ollama without sharing any code with them.</p></li>
</ul>

<hr>

## Table of contents

<details>
<summary><strong>Click to expand the table of contents</strong></summary>

<ul>
<li><a href="#status">Status</a></li>
<li><a href="#what-pickle-does">What pickle does</a>
  <ul>
  <li><a href="#supported-tensor-types">Supported tensor types</a></li>
  <li><a href="#supported-forward-pass">Supported forward pass</a></li>
  <li><a href="#tokenizer">Tokenizer</a></li>
  </ul>
</li>
<li><a href="#performance">Performance</a></li>
<li><a href="#quick-start">Quick start</a>
  <ul>
  <li><a href="#1-build">1. Build</a></li>
  <li><a href="#2-run-the-selftest">2. Run the selftest</a></li>
  <li><a href="#3-inspect-a-model">3. Inspect a model</a></li>
  <li><a href="#4-generate-text">4. Generate text</a></li>
  <li><a href="#5-chat-interactively">5. Chat interactively</a></li>
  <li><a href="#6-encode--decode-tokens">6. Encode / decode tokens</a></li>
  <li><a href="#7-benchmark">7. Benchmark</a></li>
  <li><a href="#8-dequantize-a-tensor">8. Dequantize a tensor</a></li>
  </ul>
</li>
<li><a href="#requirements">Requirements</a></li>
<li><a href="#building-and-tuning">Building and tuning</a>
  <ul>
  <li><a href="#make-targets">Make targets</a></li>
  <li><a href="#build-knobs">Build knobs</a></li>
  </ul>
</li>
<li><a href="#cli-reference">CLI reference</a>
  <ul>
  <li><a href="#selftest">selftest</a></li>
  <li><a href="#info">info</a></li>
  <li><a href="#infer">infer</a></li>
  <li><a href="#chat">chat</a></li>
  <li><a href="#bench">bench</a></li>
  <li><a href="#tokens">tokens</a></li>
  <li><a href="#dequant">dequant</a></li>
  <li><a href="#version">version</a></li>
  <li><a href="#common-options">Common options</a></li>
  <li><a href="#exit-codes">Exit codes</a></li>
  </ul>
</li>
<li><a href="#architecture">Architecture</a>
  <ul>
  <li><a href="#the-four-layers">The four layers</a></li>
  <li><a href="#same-source-two-builds">Same source, two builds</a></li>
  <li><a href="#repository-layout">Repository layout</a></li>
  </ul>
</li>
<li><a href="#why">Why</a></li>
<li><a href="#relationship-to-lestraos">Relationship to lestraOS</a></li>
<li><a href="#roadmap">Roadmap</a></li>
<li><a href="#license">License</a></li>
<li><a href="#related">Related</a></li>
</ul>

</details>

<hr>

## Status

<p><strong>v1.0.0 stable.</strong> The Llama-family forward pass is verified
end-to-end both in-kernel (lestraOS boot-time selftest, KE-28) and on the
host (<code>./pickle selftest</code>, <code>./pickle infer</code>,
<code>./pickle chat</code>, <code>./pickle bench</code>,
<code>./pickle tokens</code>). The single-sequence path, the BPE tokenizer,
and the full CLI are now frozen for the 1.x line &mdash; see
<a href="#stability">stability guarantees</a> and
<a href="CHANGELOG.md"><code>CHANGELOG.md</code></a>.</p>

<p>Continuous integration builds and selftests pickle on every push and pull
request &mdash; see <a href=".github/workflows/ci.yml"><code>.github/workflows/ci.yml</code></a>.
For build/test guidance see <a href="CONTRIBUTING.md"><code>CONTRIBUTING.md</code></a>;
for security reports see <a href="SECURITY.md"><code>SECURITY.md</code></a>.</p>

<p>Roadmap progress (see <a href="docs/ROADMAP.md"><code>docs/ROADMAP.md</code></a>):</p>

<table>
<thead>
<tr>
<th>Milestone</th>
<th>Status</th>
</tr>
</thead>
<tbody>
<tr>
<td>v0.1 &mdash; GGUF parser + basic dequant (F32/F16/Q8_0/Q4_0/Q4_1/Q5_0/Q5_1)</td>
<td><strong>Done</strong></td>
</tr>
<tr>
<td>v0.2 &mdash; K-quant family (Q4_K/Q5_K/Q6_K/Q8_K/Q2_K/Q3_K)</td>
<td><strong>Done</strong></td>
</tr>
<tr>
<td>v0.3 &mdash; Llama forward pass (RMSNorm, GQA+RoPE, SwiGLU, greedy/temp sampling)</td>
<td><strong>Done</strong></td>
</tr>
<tr>
<td>v0.4 &mdash; Llama BPE tokenizer with byte-fallback + streaming chat REPL</td>
<td><strong>Done</strong></td>
</tr>
<tr>
<td>v0.5 &mdash; Batched prefill + continuous batching</td>
<td>Next</td>
</tr>
<tr>
<td>v1.0 &mdash; First stable release (CLI + API frozen for the 1.x line)</td>
<td><strong>Done</strong></td>
</tr>
</tbody>
</table>

<a id="what-pickle-does"></a>

## What pickle does

<p>Pickle takes a <code>.gguf</code> file on disk and either (a) prints what
is inside it, (b) runs a forward pass to generate text, or (c) lets you talk
to the model interactively. Nothing else. There is no server, no HTTP API,
no GPU, no plugin system. It is one C program you run from a shell.</p>

<a id="supported-tensor-types"></a>

### Supported tensor types

<p>Every tensor type below is dequantized to F32 by an independent,
hand-written routine (no ggml dispatch table):</p>

<dl>
<dt>Full dequant</dt>
<dd><code>F32</code>, <code>F16</code>, <code>Q8_0</code>, <code>Q4_0</code>, <code>Q4_1</code>, <code>Q5_0</code>, <code>Q5_1</code></dd>
<dt>Simplified K-quants</dt>
<dd><code>Q4_K</code>, <code>Q5_K</code>, <code>Q6_K</code>, <code>Q8_K</code>, <code>Q2_K</code>, <code>Q3_K</code></dd>
</dl>

<a id="supported-forward-pass"></a>

### Supported forward pass

<p>Llama-family architectures (Llama, Llama 2, Llama 3, Mistral, Qwen2,
Phi3-style):</p>

<ul>
<li>RMSNorm (with per-layer eps read from GGUF metadata)</li>
<li>Grouped-query attention with RoPE in the NeoX convention</li>
<li>SwiGLU feed-forward network</li>
<li>Optional logit softcapping (Gemma-style)</li>
<li>Optional tied word embeddings</li>
<li>Greedy and temperature / nucleus (top-p) sampling</li>
</ul>

<a id="tokenizer"></a>

### Tokenizer

<p>The host build ships a real <strong>Llama BPE tokenizer</strong> read
directly from the <code>tokenizer.ggml.*</code> GGUF metadata, with
SentencePiece <code>&#9619;</code> (U+2581 space marker) handling and
byte-fallback. It supports BOS prepend, special-token awareness, and
both encode (<var>text</var> &rarr; token ids) and decode (token ids
&rarr; text) directions, exposed via the <code>tokens</code> subcommand
and used internally by <code>infer</code> / <code>chat</code> /
<code>bench</code>.</p>

<p>The <code>selftest</code> subcommand uses a character-level tokenizer on
the embedded demo model &mdash; sufficient to exercise the forward pass
end to end without shipping a full vocab.</p>

<hr>

<a id="performance"></a>

## Performance

<p>On a 2-core host with AVX-512 VNNI, running
<code>TinyLlama-1.1B-Chat Q4_K_M</code> (636 MiB on disk), the host fast
path delivers:</p>

<table>
<thead>
<tr>
<th>Metric</th>
<th>Baseline (v0.3)</th>
<th>Optimized (v0.4)</th>
<th>Change</th>
</tr>
</thead>
<tbody>
<tr><td>Decode (tok/s)</td><td>7.7</td><td>11.5</td><td><strong>+49%</strong></td></tr>
<tr><td>Prefill (tok/s)</td><td>8.4</td><td>11.8</td><td>+40%</td></tr>
<tr><td>32-token generation wall time</td><td>7.3 s</td><td>3.2 s</td><td><strong>&minus;56%</strong></td></tr>
<tr><td>636 MiB model startup</td><td>~5 s</td><td>&lt; 0.1 s</td><td>instant</td></tr>
</tbody>
</table>

<p>The five optimizations behind this (all host-only; the kernel soft-float
path is untouched) landed in commits
<a href="https://github.com/lee-muriethi-kingori/picklestramk/commit/f4e22b8"><code>f4e22b8</code></a>,
<a href="https://github.com/lee-muriethi-kingori/picklestramk/commit/954be76"><code>954be76</code></a>,
<a href="https://github.com/lee-muriethi-kingori/picklestramk/commit/2aec00a"><code>2aec00a</code></a>,
and
<a href="https://github.com/lee-muriethi-kingori/picklestramk/commit/0e857ed"><code>0e857ed</code></a>:</p>

<ol>
<li><p><strong><code>mmap</code> zero-copy loader.</strong> The whole GGUF file
is <code>mmap</code>&rsquo;d once and every tensor data pointer is patched to
point straight into the mapping. Zero <code>malloc</code>, zero
<code>fread</code>, zero copy. Startup for a 636 MiB model drops from
~5 s to under 0.1 s. Same idea as llama.cpp&rsquo;s <code>ggml</code> mmap
backend, implemented independently.</p></li>
<li><p><strong>Pre-allocated working buffers.</strong> 14 per-token scratch
buffers (<code>x</code>, <code>q</code>, <code>k</code>, <code>v</code>,
<code>scores</code>, gate/up/act/down, &hellip;) are allocated once in
<code>pickle_fast_state_init</code> with <code>posix_memalign</code> and
reused for every token. Eliminates 13 <code>malloc</code> + 13
<code>free</code> per token.</p></li>
<li><p><strong>Thread-local x-quant scratch.</strong> The AVX-512 VNNI
Q4_K / Q6_K kernels reuse per-thread <code>tls_x_int</code> /
<code>tls_x_scale</code> / <code>tls_sum_x16</code> buffers grown on demand,
eliminating 924 allocator calls per token (154 matmuls &times; 6).</p></li>
<li><p><strong>AVX-512 vectorized attention.</strong> The score dot-product
and V weighted sum are unrolled across 4 <code>zmm</code> FMAs per position
(vs 64 scalar FMAs), with OpenMP parallelism across heads.</p></li>
<li><p><strong>Software prefetching</strong> in the Q4_K / Q6_K VNNI kernels
&mdash; <code>_mm_prefetch</code> of the next two blocks into L2 while the
current block runs, hiding ~100 ns of DRAM latency per block. The single
biggest win (+28% on its own).</p></li>
</ol>

<p>Quantized matmul has a 3-tier dispatch per type, chosen at compile time
from compiler defines so the same binary runs everywhere and just gets
faster on newer hardware:</p>

<pre><code>__AVX512VNNI__  -&gt;  hand-rolled VNNI vpdpbusd kernel   (fastest)
__AVX512F__     -&gt;  inline 16-wide float FMA kernel
#else           -&gt;  scalar / OpenMP-simd fallback
</code></pre>

<p><code>Q4_K</code> and <code>Q6_K</code> ship VNNI kernels;
<code>Q8_0</code> / <code>Q4_0</code> / <code>Q5_*</code> use the FMA / simd
path.</p>

<hr>

<a id="quick-start"></a>

## Quick start

<p>This section walks through every common task, with the exact command and
the kind of output you should see. Copy-paste friendly.</p>

<a id="1-build"></a>

### 1. Build

<pre><code>make
</code></pre>

<p>This produces two binaries at the repository root:</p>

<dl>
<dt><code>./pickle</code></dt>
<dd>The CLI: <code>selftest</code> | <code>info</code> | <code>infer</code> | <code>chat</code> | <code>bench</code> | <code>tokens</code> | <code>dequant</code>.</dd>
<dt><code>./pickle_selftest</code></dt>
<dd>A tiny <code>main()</code> that calls <code>pickle_selftest()</code> only &mdash; mirrors the in-kernel boot-time selftest path.</dd>
</dl>

<a id="2-run-the-selftest"></a>

### 2. Run the selftest

<pre><code>./pickle selftest
</code></pre>

<p>Parses the embedded 4 KiB demo GGUF, runs one Llama forward pass, and
prints a deterministic result. Expected output:</p>

<pre><samp>pickle: selftest arch=llama L=1 H=2 HK=1 D=4 HD=8 VS=8
pickle: selftest OK, next token = 5
selftest rc=0 token=5</samp></pre>

<p>If you see <code>rc=0</code> and a token id, the parser, dequantizer,
and forward pass all work on your machine. No model file is needed for
this step &mdash; the demo model is compiled into the binary.</p>

<a id="3-inspect-a-model"></a>

### 3. Inspect a model

<pre><code>./pickle info &lt;model.gguf&gt;
</code></pre>

<p>Prints the detected architecture and the first tensors. For example,
against <code>TinyLlama-1.1B-Chat-Q4_K_M.gguf</code>:</p>

<pre><samp>arch             = llama
n_layers         = 22
hidden_dim       = 2048
n_heads          = 32
n_kv_heads       = 4
head_dim         = 64
intermediate_dim = 5632
vocab_size       = 32003
max_seq_len      = 2048
rope_theta       = 10000
rope_freq_scale  = 1
rms_eps          = 1e-05
tie_word_embeds  = 0
norm_type        = 0
act_type         = 0
rope_type        = 1
tensor_count     = 201
total_tensor_bytes = 667087152  (636.2 MiB)
--- first 16 tensors ---
  [  0] token_embd.weight                Q4_K  dims=[2048,32003] n_elems=65542144  bytes=36867456
  [  1] blk.0.attn_q.weight              Q4_K  dims=[2048,2048]   n_elems=4194304   bytes=2359296
  ...</samp></pre>

<a id="4-generate-text"></a>

### 4. Generate text

<pre><code>./pickle infer &lt;model.gguf&gt; "&lt;prompt&gt;" [N] [options]
</code></pre>

<p>Generates <code>N</code> tokens (default 20) from <code>&lt;prompt&gt;</code>
using the BPE tokenizer, and prints the generated text followed by a
timing summary. Example:</p>

<pre><code>./pickle infer model.gguf "The capital of France is" --max 8 --temp 0
</code></pre>

<pre><samp> a city in the department of the Lo
--- 6 prompt tokens, 8 generated in 468.3 ms prefill + 95.8 ms/tok decode ---</samp></pre>

<p><code>--temp 0</code> means greedy (deterministic). Raise it and add
<code>--top-p</code> for sampling.</p>

<a id="5-chat-interactively"></a>

### 5. Chat interactively

<pre><code>./pickle chat &lt;model.gguf&gt; [options]
</code></pre>

<p>Opens a streaming REPL. You type a prompt, press <kbd>Enter</kbd>, and
tokens stream back one at a time as they are decoded. Type
<code>:quit</code> (or press <kbd>Ctrl</kbd>+<kbd>D</kbd>) to exit.</p>

<pre><samp>pickle chat -- model: model.gguf (llama, 22L). Type :quit to exit.

&gt;&gt;&gt; Hello there
I'm, I am a new
&gt;&gt;&gt; :quit</samp></pre>

<p>Default cap is 256 generated tokens per turn; override with
<code>--max</code>.</p>

<a id="6-encode--decode-tokens"></a>

### 6. Encode / decode tokens

<p>The <code>tokens</code> subcommand exposes the BPE tokenizer directly,
which is useful for debugging prompts and verifying that a model file
contains a usable vocab.</p>

<p><strong>Encode</strong> text to token ids:</p>

<pre><code>./pickle tokens &lt;model.gguf&gt; encode "Hello world"
</code></pre>

<pre><samp>2 tokens:
  [  0] id=15043    "&#9619;Hello"
  [  1] id=3186     "&#9619;world"</samp></pre>

<p><strong>Decode</strong> a list of token ids back to text:</p>

<pre><code>./pickle tokens &lt;model.gguf&gt; decode 15043 3186
</code></pre>

<a id="7-benchmark"></a>

### 7. Benchmark

<pre><code>./pickle bench &lt;model.gguf&gt; ["&lt;prompt&gt;"] [N] [options]
</code></pre>

<p>Runs a prefill + decode pass and prints a throughput report. Default is
64 generated tokens. Example:</p>

<pre><code>./pickle bench model.gguf "The quick brown fox" --max 16 --temp 0
</code></pre>

<pre><samp>model         : model.gguf
arch          : llama  L=22 H=32 HK=4 D=64 HD=2048 VS=32003
prompt tokens : 6  (prefill in 466.1 ms = 12.9 tok/s)
generated     : 16  (avg 86.90 ms/tok = 11.5 tok/s decode)</samp></pre>

<a id="8-dequantize-a-tensor"></a>

### 8. Dequantize a tensor

<pre><code>./pickle dequant &lt;model.gguf&gt; &lt;tensor-name&gt;
</code></pre>

<p>Prints the first 32 floats of <code>&lt;tensor-name&gt;</code> after
dequantization to F32. Useful for cross-checking a dequantizer against a
reference implementation:</p>

<pre><code>./pickle dequant model.gguf token_embd.weight
</code></pre>

<hr>

<a id="requirements"></a>

## Requirements

<ul>
<li><p>A C compiler: <code>cc</code>, <code>gcc</code>, or <code>clang</code>.</p></li>
<li><p><code>make</code>.</p></li>
<li><p>Python 3 &mdash; <em>only</em> if you want to regenerate the embedded
demo GGUF via <code>tools/make_tiny_gguf.py</code>. Not required to build
or run any subcommand.</p></li>
<li><p>OpenMP &mdash; optional but on by default; gives multi-threaded matmul
that scales with core count.</p></li>
<li><p>AVX-512 &mdash; auto-detected from <code>-march=native</code>. The FMA /
simd fallback still runs on older hardware.</p></li>
</ul>

<hr>

<a id="building-and-tuning"></a>

## Building and tuning

<a id="make-targets"></a>

### Make targets

<table>
<thead>
<tr><th>Command</th><th>What it does</th></tr>
</thead>
<tbody>
<tr><td><code>make</code> / <code>make all</code></td><td>Build <code>./pickle</code> and <code>./pickle_selftest</code></td></tr>
<tr><td><code>make pickle</code></td><td>Build only <code>./pickle</code></td></tr>
<tr><td><code>make pickle_selftest</code></td><td>Build only <code>./pickle_selftest</code></td></tr>
<tr><td><code>make test</code></td><td>Build, then run <code>./pickle_selftest</code> and <code>./pickle selftest</code></td></tr>
<tr><td><code>make bench</code></td><td>Build, then run the embedded demo selftest</td></tr>
<tr><td><code>make clean</code></td><td>Remove binaries and <code>src/*.o</code></td></tr>
</tbody>
</table>

<p><code>CFLAGS</code>, <code>LDFLAGS</code>, <code>LDLIBS</code>, and
<code>CC</code> are all overridable on the make command line or via the
environment.</p>

<a id="build-knobs"></a>

### Build knobs

<table>
<thead>
<tr><th>Flag</th><th>Effect</th></tr>
</thead>
<tbody>
<tr><td><code>make PICKLE_NO_NATIVE=1</code></td><td>Do not use <code>-march=native</code> (reproducible / cross-architecture builds; FMA/simd fallback still runs)</td></tr>
<tr><td><code>make PICKLE_NO_OMP=1</code></td><td>Disable OpenMP threading (single-threaded matmul)</td></tr>
<tr><td><code>make CFLAGS=-O0</code></td><td>Debug build</td></tr>
</tbody>
</table>

<p><code>-march=native</code> is on by default so the compiler can emit
AVX-512 VNNI when the CPU supports it.</p>

<p><strong>Fresh-checkout build and verify:</strong></p>

<pre><code>make clean &amp;&amp; make
make test
make bench
</code></pre>

<p><strong>Regenerate the tiny demo model</strong> (writes
<code>/tmp/pickle_demo.gguf</code> and refreshes
<code>src/pickle_demo_gguf.c</code>, after which you should re-run
<code>make</code>):</p>

<pre><code>python3 tools/make_tiny_gguf.py
make
./pickle info /tmp/pickle_demo.gguf
./pickle infer /tmp/pickle_demo.gguf "abc" 5
</code></pre>

<p><strong>Using a real model:</strong> download any Llama-family Q4_K_M GGUF
(TinyLlama-1.1B, Qwen2-0.5B, Phi-2, etc.) and point <code>./pickle infer</code>
at it. The host build uses <code>mmap</code> by default, so a 636 MiB model
loads in well under a second.</p>

<hr>

<a id="cli-reference"></a>

## CLI reference

<p>Run <code>./pickle</code> with no arguments (or <code>./pickle --help</code>)
to see the usage banner:</p>

<pre><samp>pickle -- a from-scratch GGUF model loader and inference engine

Usage:
  pickle selftest                                 Run the embedded selftest
  pickle info &lt;model.gguf&gt;                        Print model architecture + tensors
  pickle infer &lt;model.gguf&gt; "&lt;prompt&gt;" [N] [opts]  Generate N tokens (default 20)
  pickle chat  &lt;model.gguf&gt; [opts]                Interactive REPL (streaming)
  pickle bench &lt;model.gguf&gt; ["&lt;prompt&gt;"] [N] [opts] Benchmark prefill + decode tok/s
  pickle tokens &lt;model.gguf&gt; encode "&lt;text&gt;"      BPE-encode text -&gt; token ids
  pickle tokens &lt;model.gguf&gt; decode &lt;id...&gt;        Decode token ids -&gt; text
  pickle dequant &lt;model.gguf&gt; &lt;tensor&gt;            Print first 32 floats of &lt;tensor&gt;
  pickle version | --version | -V                  Print version and exit

Options (infer / chat / bench):
  --temp T     Temperature (default 0 = greedy)
  --top-p P    Nucleus sampling threshold (default 0.9)
  --seed S     RNG seed (default 0xC0FFEE)
  --max N      Max new tokens (default 20 infer / 256 chat / 64 bench)
  --no-bos     Don't prepend BOS token</samp></pre>

<details>
<summary><strong>Per-subcommand detail</strong></summary>

<a id="selftest"></a>

#### selftest

<pre><code>pickle selftest
</code></pre>

<p>No arguments. Parses the embedded 4 KiB demo GGUF (1 layer, 8-dim,
16-vocab F32 Llama), runs one forward pass, and prints the sampled next
token. This is the same code path that lestraOS runs at boot time as
KE-28. Exits <code>0</code> on success.</p>

<a id="info"></a>

#### info

<pre><code>pickle info &lt;model.gguf&gt;
</code></pre>

<p>Reads the GGUF header and metadata, prints the detected architecture
(<code>arch</code>, layer count, head / KV-head / head-dim counts,
intermediate dim, vocab size, max sequence length, RoPE theta and scale,
RMS eps, tied-embeddings flag, norm / act / rope type) and the first 16
tensors with name, dtype, dims, element count, and byte size. Does
<em>not</em> dequantize or run a forward pass.</p>

<a id="infer"></a>

#### infer

<pre><code>pickle infer &lt;model.gguf&gt; "&lt;prompt&gt;" [N] [options]
</code></pre>

<p>BPE-encodes the prompt (prepending BOS unless <code>--no-bos</code>),
runs prefill, then samples <code>N</code> tokens (default 20) one at a
time and prints the decoded text, followed by a one-line timing summary
(prefill ms, decode ms/tok). <code>--temp 0</code> is greedy and
deterministic.</p>

<a id="chat"></a>

#### chat

<pre><code>pickle chat &lt;model.gguf&gt; [options]
</code></pre>

<p>Streaming interactive REPL. Reads one prompt per line from stdin,
generates up to <code>--max</code> tokens (default 256) per turn, and
streams each token to stdout as soon as it is decoded. Type
<code>:quit</code> or send EOF (<kbd>Ctrl</kbd>+<kbd>D</kbd>) to exit.
The KV cache is preserved across turns within a session so the model
remembers the conversation.</p>

<a id="bench"></a>

#### bench

<pre><code>pickle bench &lt;model.gguf&gt; ["&lt;prompt&gt;"] [N] [options]
</code></pre>

<p>Runs one prefill + decode pass (default 64 generated tokens) and prints
a throughput report: prompt token count, prefill time and tok/s, generated
token count, and average decode ms/tok and tok/s. Use this to compare
quantizations, thread counts, or hardware.</p>

<a id="tokens"></a>

#### tokens

<pre><code>pickle tokens &lt;model.gguf&gt; encode "&lt;text&gt;"
pickle tokens &lt;model.gguf&gt; decode &lt;id&gt; [&lt;id&gt; ...]
</code></pre>

<p>Exposes the BPE tokenizer directly. <code>encode</code> prints each
token id with its piece text (SentencePiece <code>&#9619;</code> shown
explicitly). <code>decode</code> takes one or more numeric token ids and
prints the decoded text. Handy for inspecting how a prompt is split, or
for verifying that a model file ships a usable vocab.</p>

<a id="dequant"></a>

#### dequant

<pre><code>pickle dequant &lt;model.gguf&gt; &lt;tensor-name&gt;
</code></pre>

<p>Locates <code>&lt;tensor-name&gt;</code> in the tensor-info table, reads
its raw bytes from the <code>mmap</code>&rsquo;d region, dequantizes to F32,
and prints the first 32 floats. Useful for cross-checking a dequantizer
against a reference implementation (e.g. comparing Q4_K output against
llama.cpp bit-for-bit on the same tensor).</p>

<a id="version"></a>

#### version

<pre><code>pickle version
pickle --version
pickle -V
</code></pre>

<p>Any of the three forms prints the pickle version string (e.g.
<samp>pickle 1.0.0</samp>) and exits <code>0</code>. The string is generated
from <a href="src/version.h"><code>src/version.h</code></a> and is the single
source of truth for which release you are running. It matches the most recent
<code>v*</code> git tag.</p>

</details>

<a id="common-options"></a>

### Common options

<p>The <code>--temp</code>, <code>--top-p</code>, <code>--seed</code>,
<code>--max</code>, and <code>--no-bos</code> flags are accepted by
<code>infer</code>, <code>chat</code>, and <code>bench</code>. They may
appear in any order after the positional arguments.</p>

<table>
<thead>
<tr><th>Flag</th><th>Default</th><th>Meaning</th></tr>
</thead>
<tbody>
<tr><td><code>--temp T</code></td><td><code>0</code> (greedy)</td><td>Sampling temperature. <code>0</code> = greedy argmax; <code>&gt;0</code> = sample from the softmax distribution scaled by <code>1/T</code>.</td></tr>
<tr><td><code>--top-p P</code></td><td><code>0.9</code></td><td>Nucleus sampling threshold. Only the smallest set of tokens whose cumulative probability reaches <code>P</code> is sampled from. Ignored when <code>--temp 0</code>.</td></tr>
<tr><td><code>--seed S</code></td><td><code>0xC0FFEE</code></td><td>RNG seed for reproducible sampling. <code>0</code> = time-seeded.</td></tr>
<tr><td><code>--max N</code></td><td>20 infer / 256 chat / 64 bench</td><td>Maximum number of new tokens to generate.</td></tr>
<tr><td><code>--no-bos</code></td><td>off</td><td>Do not prepend the model&rsquo;s BOS token to the prompt.</td></tr>
</tbody>
</table>

<a id="exit-codes"></a>

### Exit codes

<table>
<thead>
<tr><th>Code</th><th>Meaning</th></tr>
</thead>
<tbody>
<tr><td><code>0</code></td><td>Success.</td></tr>
<tr><td><code>1</code></td><td>Error (bad arguments, unreadable file, unsupported tensor type, allocation failure, etc.). An error message is printed to stderr.</td></tr>
</tbody>
</table>

<hr>

<a id="architecture"></a>

## Architecture

<a id="the-four-layers"></a>

### The four layers

<p>Pickle is four layers. Each higher layer may call the layer below it,
never the reverse, and never sideways. The boundary lines are enforced by
link-time discipline (each layer is its own translation unit) and by the
<code>PICKLE_KERNEL</code> preprocessor toggle in <code>pickle.h</code>.</p>

<pre><code> +----------------------------------------------------------------+
 |  pickle_cli   (src/pickle_cli.c)                                |
 |  argv parsing, subcommands: selftest | info | infer | chat |   |
 |    bench | tokens | dequant                                     |
 +----------------------------------------------------------------+
 |  pickle_host  (src/pickle_host.c)                  [host only]  |
 |  POSIX shim: FILE*-based pickle_io_t, malloc allocator,        |
 |    pickle_load_from_file() AND pickle_load_from_file_mmap()    |
 |    (zero-copy mmap), pickle_run_prompt() / pickle_run_chat()   |
 +----------------------------------------------------------------+
 |  pickle_fast  (src/pickle_fast.c + src/pickle_tokenizer.c +    |
 |                src/pickle_fast.h)                  [host only]  |
 |  native float math (SSE/AVX/AVX-512 auto-vectorised),        |
 |    Llama BPE tokenizer (SentencePiece '▁' + byte-fallback),    |
 |    quantized Q4_K/Q6_K AVX-512 VNNI matmul, precomputed RoPE,  |
 |    contiguous KV cache, pre-allocated per-token buffers.       |
 |    pickle_forward() dispatches here when !PICKLE_KERNEL.       |
 +----------------------------------------------------------------+
 |  pickle_core  (src/pickle.c + src/pickle_softfp.c +            |
 |                src/pickle.h)                                    |
 |  freestanding C, no libc, no syscalls. GGUF parse,             |
 |    dequantize, Llama forward pass (soft-float), greedy/temp    |
 |    sampling. All math goes through sfp_t (uint32_t IEEE-754    |
 |    bit patterns) and the sfp_*() soft-float functions --       |
 |    NEVER C float arithmetic, so it links into -mno-sse builds. |
 +----------------------------------------------------------------+
</code></pre>

<a id="same-source-two-builds"></a>

### Same source, two builds

<p>The same <code>pickle.c</code> / <code>pickle_softfp.c</code> /
<code>pickle.h</code> source compiles two ways:</p>

<ul>
<li><p><strong>In the lestraOS kernel</strong> &mdash; with
<code>-DPICKLE_KERNEL</code>. The <code>#ifdef PICKLE_KERNEL</code> blocks
at the top of each file pull in <code>&lt;lestra/types.h&gt;</code>,
<code>&lt;lestra/printk.h&gt;</code>, <code>&lt;lestra/mm.h&gt;</code> and
the kernel bump allocator is used. <code>pickle_forward()</code> falls back
to the soft-float path. This is the in-kernel build at lestraOS commit
<code>8d3300c</code> (KE-28).</p></li>
<li><p><strong>On the host</strong> (this repo) &mdash; with
<code>-UPICKLE_KERNEL</code> (the default). Those <code>#ifdef</code>
blocks pull in <code>&lt;stdint.h&gt;</code>, <code>&lt;stdio.h&gt;</code>,
<code>&lt;stdlib.h&gt;</code>, <code>&lt;string.h&gt;</code> instead,
<code>pr_info</code> is <code>#define</code>&rsquo;d to <code>printf</code>,
<code>kmalloc</code> to <code>malloc</code>, and so on. The POSIX shim
(<code>pickle_host.c</code>) provides <code>FILE*</code>-based and
<code>mmap</code>-based <code>pickle_io_t</code> callbacks and a
<code>malloc</code>-based <code>pickle_alloc_t</code>.
<code>pickle_forward()</code> dispatches to the fast path in
<code>pickle_fast.c</code>.</p></li>
</ul>

<p>No glue or <code>#ifdef _HOST_</code> is needed inside the core files
&mdash; the <code>PICKLE_KERNEL</code> toggle handles everything. The fast
path is fenced off behind <code>#ifndef PICKLE_KERNEL</code>, so the kernel
build never sees native float math, <code>mmap</code>, OpenMP, or AVX-512
intrinsics.</p>

<a id="repository-layout"></a>

### Repository layout

<pre><code>picklestramk/
+-- Makefile                       builds ./pickle and ./pickle_selftest
+-- src/                           the pickle engine source
|   +-- pickle.h                   public API (freestanding)
|   +-- pickle.c                   GGUF parse + Llama fwd pass (soft-float)
|   +-- pickle_softfp.c            IEEE-754 soft float32
|   +-- pickle_fast.h              host fast-path API
|   +-- pickle_fast.c              AVX-512 VNNI matmul + native forward
|   +-- pickle_tokenizer.c         Llama BPE tokenizer (host)
|   +-- pickle_demo_gguf.c         embedded tiny GGUF (auto-generated)
|   +-- pickle_host.c              POSIX shim (host only)
|   +-- pickle_cli.c               CLI frontend (host only)
|   +-- pickle_selftest_main.c     tiny main() for ./pickle_selftest
+-- tools/
|   +-- make_tiny_gguf.py          regenerates the embedded demo model
+-- docs/
    +-- ARCHITECTURE.md
    +-- ROADMAP.md
</code></pre>

<blockquote>
<p><strong>Why does source live under <code>src/</code> rather than
<code>pickle/</code>?</strong> Because the CLI binary is <code>./pickle</code>
at the repo root, and a POSIX filesystem cannot have both a
<code>pickle/</code> directory and a <code>pickle</code> file at the same
level (a directory entry name is unique). Putting source under
<code>src/</code> lets the verification commands <code>./pickle selftest</code>,
<code>./pickle info &hellip;</code>, <code>./pickle infer &hellip;</code>,
<code>./pickle dequant &hellip;</code> work exactly as written.</p>
</blockquote>

<hr>

<a id="why"></a>

## Why

<p>Because the lestraOS kernel disables SSE (<code>-mno-sse</code>) and has
no x87 init, so any <code>float</code> arithmetic would raise
<code>#NM</code>/<code>#UD</code>. Pickle ships its own software float32
layer (<code>pickle_softfp.c</code>, ~370 lines of integer-only IEEE-754
binary32 add/sub/mul/div/exp/tanh/sigmoid/silu/gelu/sqrt/rsqrt/sin/cos) so
the <strong>same core source</strong> runs anywhere &mdash; kernel,
embedded, anywhere. On the host, where SSE/AVX are available, the fast
path (<code>pickle_fast.c</code>) layers on top of that same core to
deliver production-grade throughput with hand-tuned AVX-512 VNNI kernels.</p>

<p>The same engine that boots inside the lestraOS kernel (verifying the
GGUF parser and Llama forward pass at boot time, with no filesystem access,
using an embedded 4 KB demo model) is the one you build with <code>make</code>
here &mdash; and the host build adds the mmap loader, BPE tokenizer, and
VNNI matmul on top.</p>

<hr>

<a id="relationship-to-lestraos"></a>

## Relationship to lestraOS

<p>lestraOS vendors the <strong>kernel-compatible</strong> half of this
repo (the core <code>pickle.c</code>, <code>pickle_softfp.c</code>,
<code>pickle.h</code>, and <code>pickle_demo_gguf.c</code>) into
<code>kernel/ai/</code> and <code>kernel/include/lestra/</code>. The fast
path (<code>pickle_fast.c</code>, <code>pickle_tokenizer.c</code>,
<code>pickle_host.c</code>, <code>pickle_cli.c</code>) is host-only and
lives exclusively in this repo.</p>

<p>lestraOS tracks this repo as a git submodule at
<code>third_party/picklestramk</code> and ships a
<code>scripts/sync_picklestramk.sh</code> that copies the kernel-compatible
sources into the kernel tree. See lestraOS&rsquo;s
<a href="https://github.com/lee-muriethi-kingori/LestraOS/blob/main/docs/PICKLESTRAMK.md"><code>docs/PICKLESTRAMK.md</code></a>
for the integration guide.</p>

<hr>

<a id="stability"></a>

## Stability guarantees (1.x line)

<p>With <strong>v1.0.0</strong> the following surfaces are frozen for the
whole 1.x line &mdash; they will not be removed or renamed, and behaviour
will not change for existing valid input without a major version bump:</p>

<ul>
<li><p><strong>The CLI.</strong> The subcommands <code>selftest</code>,
<code>info</code>, <code>infer</code>, <code>chat</code>, <code>bench</code>,
<code>tokens</code>, <code>dequant</code>, and <code>version</code>, and the
flags <code>--temp</code>, <code>--top-p</code>, <code>--seed</code>,
<code>--max</code>, <code>--no-bos</code>, <code>--version</code> / <code>-V</code>,
and <code>--help</code> / <code>-h</code>. New subcommands and flags may be
added (minor bumps); existing ones keep their names, argument order, and
defaults.</p></li>
<li><p><strong>The <code>pickle_core</code> API</strong> declared in
<a href="src/pickle.h"><code>src/pickle.h</code></a> &mdash; the model loader,
architecture descriptor, and forward-pass entry points.</p></li>
<li><p><strong>The host/fast APIs</strong> in
<a href="src/pickle_fast.h"><code>src/pickle_fast.h</code></a>,
<code>src/pickle_host.c</code>, and <code>src/pickle_tokenizer.c</code>
&mdash; the mmap loader, the BPE tokenizer, and the streaming chat loop.</p></li>
<li><p><strong>The supported tensor types</strong> (F32, F16, Q8_0, Q4_0,
Q4_1, Q5_0, Q5_1, Q4_K, Q5_K, Q6_K, Q8_K, Q2_K, Q3_K) and the
Llama-family forward pass (RMSNorm + GQA/RoPE + SwiGLU, greedy and
temperature/top-p sampling).</p></li>
</ul>

<p>Explicitly <strong>not</strong> part of the 1.0 stability surface:
batched prefill and continuous batching (tracked as v0.5 / a future 2.0),
multi-turn chat context (each <code>chat</code> prompt is independent in
1.x), and any build-time knob&rsquo;s default value other than the ones
documented above.</p>

<p>The version string reported by <code>./pickle --version</code> is the
single source of truth and is generated from
<a href="src/version.h"><code>src/version.h</code></a>. Per-release changes
are recorded in <a href="CHANGELOG.md"><code>CHANGELOG.md</code></a>.</p>

<hr>

<a id="roadmap"></a>

## Roadmap

<p>The full version plan lives in
<a href="docs/ROADMAP.md"><code>docs/ROADMAP.md</code></a>. Summary:</p>

<table>
<thead>
<tr><th>Version</th><th>Theme</th><th>Status</th></tr>
</thead>
<tbody>
<tr><td>v0.1</td><td>GGUF parser + basic dequant (F32/F16/Q8_0/Q4_0/Q4_1/Q5_0/Q5_1)</td><td>Done</td></tr>
<tr><td>v0.2</td><td>K-quant family (Q4_K/Q5_K/Q6_K/Q8_K/Q2_K/Q3_K)</td><td>Done</td></tr>
<tr><td>v0.3</td><td>Llama forward pass (RMSNorm, GQA+RoPE, SwiGLU, sampling)</td><td>Done</td></tr>
<tr><td>v0.4</td><td>Llama BPE tokenizer with byte-fallback + streaming chat</td><td>Done</td></tr>
<tr><td>v0.5</td><td>Batched prefill + continuous batching</td><td>Next</td></tr>
<tr><td>v1.0</td><td>First stable release (CLI + API frozen for the 1.x line)</td><td>Done</td></tr>
</tbody>
</table>

<hr>

<a id="license"></a>

## License

<p>MIT, Copyright (c) 2026 Lee Muriithi Kingori. See
<a href="LICENSE"><code>LICENSE</code></a>.</p>

<a id="related"></a>

## Related

<dl>
<dt>This repo</dt>
<dd><a href="https://github.com/lee-muriethi-kingori/picklestramk">https://github.com/lee-muriethi-kingori/picklestramk</a></dd>
<dt>lestraOS</dt>
<dd><a href="https://github.com/lee-muriethi-kingori/LestraOS">https://github.com/lee-muriethi-kingori/LestraOS</a> &mdash; the OS where pickle runs in-kernel. The in-kernel half of this work shipped as KE-28 (commit <code>8d3300c</code>): a boot-time selftest that parses the embedded demo GGUF and runs one Llama forward pass, printing <code>pickle: selftest OK, next token = 5</code> to the kernel console.</dd>
</dl>
