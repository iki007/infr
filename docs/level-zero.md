# Level Zero backend (`infr-ze`) — Intel Xe2, native

Status (2026-11-10): **prefill GEMMs, head-dim-256 flash attention and the ops joining them run on
Level Zero inside the Vulkan backend**, on by default on Intel Xe2 cards with Level Zero (`INFR_ZE_GEMM=0` turns it off;
`crates/infr-vulkan/src/ze.rs`): Qwen3.8-27B pp512 740 -> 1123
t/s (llama.cpp SYCL 1087), Qwen3.5-4B 4172 -> 6335, tested greedy output unchanged. Long context is
where it still trails SYCL; the attention work below narrows that gap. Backlog B102.

## Why

On the B70, the Vulkan prefill GEMM (`native_gemm_warp` `_x256`, Mesa ANV cooperative matrices)
peaks at ~50 TFLOPS and sustains ~46 (`docs/intel-b70.md`); llama.cpp's Vulkan GEMM holds ~57, and
llama.cpp SYCL (oneDNN) is 1.5–1.7x our prefill. TensorFold's Level Zero port, on the same card,
sustains 146 TFLOPS fp16 on the gate|up GEMM with nGEN-generated ISA. The question for phase 0 was
whether infr can get there without C++: OpenCL C compiled offline, driven through Level Zero.

## Shape

- **Level Zero through `dlopen`** (`crates/infr-ze/src/ffi.rs`, the `ash` analogue). The C structs
  are hand-declared from `ze_api.h` (loader 1.32) and the entry points resolved from
  `libze_loader.so.1` / `ze_loader.dll` at run time. Nothing links against oneAPI.
- **Kernels in OpenCL C, compiled by `ocloc`** at build time into native binaries embedded in the
  crate (`build.rs`, the `glslc` analogue; target `bmg-g31`, `INFR_ZE_TARGET` overrides). A kernel's
  first lines can carry `// ocloc-options:` (e.g. `-cl-intel-256-GRF-per-thread`). Without `ocloc`
  the crate builds with no kernels and the probes skip — CI and non-Intel machines are unaffected.
- OpenCL C reaches the Xe2 primitives the Vulkan path cannot: **DPAS**
  (`intel_sub_group_f16_f16_matrix_mad_k16`, 16-lane subgroups) and **2D block loads/stores**
  (`intel_sub_group_2d_block_read_*`, including the transposed and VNNI-transform forms). The
  probe's ISA (`ocloc disasm`) shows `dpas.8x8 (16|M0)`, `load_block2d.ugm.d16`/`.d16v` and
  `store_block2d`.
- The device layer (`crates/infr-ze/src/lib.rs`): device discovery by PCI address (the box has
  two identical B70s; `INFR_ZE_PCI=03:00.0` is the headless one), a context with one in-order
  immediate command list, device memory, modules, kernels, kernel timestamps.

## Phase 0 results (Arc Pro B70 `03:00.0`, compute runtime 26.40, `tests/probe.rs`)

`kernels/gemm_f16.cl`: `C[M,N] (f32) = A[M,K] (f16) · W[N,K]ᵀ (f16)`, W in weight layout.
Each 16-lane subgroup keeps a 32x64 C tile in registers (4x4 DPAS accumulators, 256-GRF mode);
A arrives as 32-row x 32-k 2D block reads, W as transposed 32-bit reads (16 rows x 8 words), which
is DPAS's B operand as loaded. No shared memory, no barriers. 8 subgroups per workgroup (128x128).

| m=512 GEMM | time (sustained, 200 runs) | TFLOPS |
| --- | --- | --- |
| 5120 -> 17408, workgroups ordered N-major | 1.583 ms | 57.6 |
| 5120 -> 17408, workgroups ordered M-major (the M tiles of one W column tile run back to back, W from L2) | **0.877 ms** | **104** |
| + 2D block prefetch two k-steps ahead | 1.086 ms | 84 (not kept) |
| 17408 -> 5120 (160 workgroups) | 1.248 ms | 73 |

For comparison, the same 5120 -> 17408 GEMM: infr Vulkan `_x256` ~2.0 ms sustained (Q4_K, its
in-shader decode included); llama.cpp Vulkan 1.55–1.60 ms in-stream. Short bursts run slower
(1.3 ms over 10 runs) while the clocks ramp.

So the native path is worth building: a ~50-line OpenCL C kernel is 2.3x our Vulkan GEMM. The
weights in a real model are quantized (Q4_K / IQ4_XS ~ 0.5–0.6 B per element against f16's 2), so
the decode path's bandwidth bound is lower, and dequantizing in registers before DPAS (or
TensorFold's int4-as-fp16-subnormal trick) is the next kernel question.

## Integration: hybrid (prefill GEMMs on Level Zero, the rest on Vulkan)

Chosen 2026-11-10 over a full fourth backend: Vulkan keeps every op, and the prefill GEMMs move to
Level Zero kernels that read the Vulkan buffers in place.

### Phase 1: interop (done, `crates/infr-vulkan/tests/ze_interop.rs`)

- **Memory**: a Vulkan buffer exported as a dma-buf (`VulkanBackend::p2p_export(.., DmaBuf)`) is
  imported with `Gpu::import_dmabuf` (`zeMemAllocDevice` + `ze_external_memory_import_fd_t`).
  Level Zero kernels read and write the Vulkan pages; 4 rounds of Vulkan-write → L0 read/write →
  Vulkan-read over 1M floats are exact, and the reverse direction too.
- **Ordering**: a Vulkan timeline semaphore (`tp_export_timeline`, `OPAQUE_FD`) is imported with
  `Gpu::import_vk_timeline` (`ZE_extension_external_semaphores`,
  `VK_TIMELINE_SEMAPHORE_FD`); `wait_semaphore` / `signal_semaphore` append GPU-side waits and
  signals to the immediate command list. Appending an unmet wait does not block the host.
- **fd ownership**: the Intel compute runtime (26.40) closes the fd it imports, for memory and
  for semaphores (measured: the descriptor is gone after a successful import). The infr-ze imports
  take an `OwnedFd`; callers pass a duplicate of the exporter's fd.

Per-handoff cost (B70 `03:00.0`, 200 rounds; one round is a 16 KB Vulkan copy, a 4096-element L0
add, a 16 KB Vulkan copy — two cross-API handoffs around ~µs of work):

| ordering | per round |
| --- | --- |
| Vulkan only: the same two submits chained by the timeline, no L0 | 74 µs |
| Vulkan → L0 → Vulkan, timeline semaphores (GPU-side) | 156–164 µs (63 µs of it host issue, overlapped) |
| Vulkan → L0 → Vulkan, host fences | 270–310 µs |

So an L0 segment between two Vulkan submits costs ~80 µs per round trip over a plain dependent
Vulkan resubmit — kernel-driver scheduling of a cross-queue dependency, not work. For 27B prefill
(64 layers), offloading every GEMM naively is ~8 handoffs per layer ≈ 512 per ubatch ≈ 20–30 ms
against ~690 ms per 512 tokens today; moving the norms/elementwise ops between GEMMs into L0 later
cuts that to ~2 per layer.

### Phase 2: quantized DPAS GEMMs (Q4_K, Q5_K, IQ4_XS done)

The 27B (UD-Q4_K_M) GEMM work by weight format: IQ4_XS 36%, Q5_K 28%, Q4_K 25%, Q6_K 3.6%, the
rest under 3% each; the 4B Q4_K_M is nearly all Q4_K (+ Q6_K).

`gemm_q4k` (`kernels/gemm_kquant.cl`): the f16 probe's tiling with the B operand dequantized in registers. Each lane owns two
W rows (DPAS's B operand is one column per lane) and turns 16 nibbles into 8 dwords of half
k-pairs: mask, widen bytes to words, `| 0x6400` (the half 1024 + q), `- 1024`, `* s + m`.

What it took (m=512, 5120 -> 17408, sustained):

| variant | time | TFLOPS | non-DPAS instr. per DPAS |
| --- | --- | --- | --- |
| OpenCL C `half16` dequant (`convert_half16(as_uchar16(..))`, `fma`) | 1.551 ms | 58.9 | ~21 |
| C^T = W·A^T (W as DPAS's A operand, whose layout is OpenCL's per-lane `half8`) | 1.812 ms | 50.4 | ~17 (the structure alone: 1.36 ms) |
| **vISA inline asm for the B operand** | **0.935 ms** | **97.6** | ~7 |
| bound: same kernel, no dequant (wrong values) | 0.800 ms | 114 | — |

OpenCL C keeps a per-lane `half16` de-interleaved in registers (element e of all 16 lanes in a
row), while DPAS's B operand wants each lane's two k-values packed in one dword; IGC moves every
value out and back. Inline vISA (`__asm__` with `"rw"` operands and `.decl ... alias`) computes the
packed rows directly: `mov (32) uw <- ub<4;2,1>` widens two nibble bytes into a dword's halves.
Two IGC inline-asm rules learnt: 32-wide ops in a SIMD16 kernel need `M1_NM` (the upper 16
channels are otherwise masked off), and there is no early-clobber constraint, so intermediates go
to asm-local `.decl` registers and only the last instruction writes the output.

Other shapes (m=512): 17408 -> 5120 1.070 ms (85 TFLOPS), 5120 -> 12288 0.577 ms (112),
5120 -> 6144 0.323 ms (100). m=2048 5120 -> 17408 is 4.85 ms (75): the 21 MB A no longer stays in
L2 across the N tiles — a packet-ordered grid (all N tiles of a 512-row M packet first) is the fix.
Against Vulkan `_x256` Q4_K (1.62 ms / ~2.19 ms for the two FFN shapes) this is 1.7-2x.

`gemm_q5k` shares the body (`gemm_kq`, specialised by a constant format): Q5_K's fifth bit is
`((qh >> sb) & 0x01010101) << 4` OR-ed into the nibble bytes before `deq_b`.

`gemm_iq4xs` (`kernels/gemm_iq4.cl`): the IQ4_NL codebook through a byte-pair table in shared
memory (`lut[a | b << 4] = half2(kv[a], kv[b])`, one gather per B row), then the sub-block scale in
vISA (`scale_b`). A degree-4 polynomial for the codebook misses by up to 0.476 — too close to the
rounding edge in f16. A sparse table indexed by the nibbles in place (`lut[a | b << 8]`, two ALU
ops per row instead of five) ran 4% slower: entries 1 KB apart share SLM banks; the gathers are
the limit.

| m=512, 5120 -> 17408, sustained | L0 | TFLOPS | Vulkan `_x256` |
| --- | --- | --- | --- |
| f16 (reference) | 0.881 ms | 104 | — |
| Q4_K | 0.925 ms | 99 | 1.62 ms |
| Q5_K | 0.856 ms | 107 | 1.77 ms |
| IQ4_XS | 1.291 ms | 71 | 1.78 ms |

### Phase 3: the engine path (`crates/infr-vulkan/src/ze.rs`; default on since 2026-11-10, `INFR_ZE_GEMM=0` off)

- **Weights**: with the knob on (and the card Intel Xe2 with dma-buf export, external timeline
  semaphores and `VK_EXT_pci_bus_info`), the resident weight blocks are dedicated exportable
  allocations (`alloc_exportable`, charged to the VRAM budget). Level Zero imports each block once,
  keyed by a process-unique `export_id`, and addresses a tensor as base + offset.
- **Activations**: an *exchange arena* (one exportable block, bump-allocated per forward) holds the
  `_x256` f16 A copies (`lin_a16_x256`, which the producers' `-DY16` builds already write) and the
  outputs of the offloaded Linears. They are ordinary `BdaSub` views, so every Vulkan op reads and
  writes them unchanged.
- **Ordering**: `execute_static` ends the segment at an offloaded Linear with a submit signalling
  the timeline `v` (`Recorder::finish_nowait_signal`), appends Level Zero's wait `v` / GEMM / signal
  `v + 1`, and opens the next segment waiting `v + 1` (`Recorder::wait_timeline`). Nothing waits on
  the host beyond the existing two-segment window. The offloaded Linear's residual `Add` is unfused.
- **What runs there**: Q4_K / Q5_K / IQ4_XS Linears with m >= 256, N % 64 == 0, K % 256 == 0 and an
  `Internal` destination; every other op (and dense streaming) stays as before.
- **The bug the first run hit**: the Level Zero immediate command list was created without
  `ZE_COMMAND_QUEUE_FLAG_IN_ORDER`, so a semaphore signal appended after a kernel could fire before
  the kernel finished: a 1 ms GEMM's consumer read a half-written output (garbage text; the small
  shapes of the first test passed). `tests/ze_linear.rs` (Linear + residual Add at the 4B gate/up
  shape, every format, on vs off vs host) now covers it.

| pp512, `Vulkan1`, 3 reps | Vulkan only | `INFR_ZE_GEMM=1` | llama.cpp Vulkan | llama.cpp SYCL |
| --- | --- | --- | --- | --- |
| Qwen3.8-27B UD-Q4_K_M | 740 | **1013** | 746 | 1087 |
| Qwen3.5-4B Q4_K_M | 4172 | **5138** | 3165 | 7031 |

Each offloaded GEMM is launched as 256-row packets (one workgroup row each): a packet's A stays in
L2 across all N tiles. Q4_K, alone: m=512 5120 -> 17408 0.93 ms as one launch, 2 x 0.42 as
packets; 17408 -> 5120 1.07 vs 2 x 0.41 (85 -> 112 TFLOPS). End to end it moves the 27B pp2048
(m=1024 ubatches) 967 -> 1028 t/s (Vulkan only 739) and leaves pp512 at ~1010.

A run of offloaded Linears on the same input (qkv + z, gate + up, q/k/v) is one Level Zero chain:
one wait, the GEMMs back to back, one signal — no empty Vulkan segment and two handoffs fewer per
link. 27B pp512 1008 -> 1037, pp2048 1028 -> 1047; 4B pp512 5138 -> 5242 (output unchanged).

**Whole runs on Level Zero.** `kernels/elementwise.cl` adds `rmsnorm`, `swiglu`, `add` and
`to_f16` (the arithmetic of the Vulkan shaders they stand in for, with the next GEMM's f16 copy as
an optional output). `ze_plan` marks every L0-capable op (offloadable Linears, f32 `Add`,
`RmsNorm` with an exportable weight, SwiGLU `GatedActFused`) and keeps each run of consecutive ones
that holds a GEMM: a layer's out-projection -> residual add -> norm -> gate|up -> SwiGLU -> down ->
add -> next norm -> input projections is one chain. The residual stream is the seam's bound
buffer, not scratch, so a bound f32 activation an L0 op touches is *shadowed* in the exchange arena
for the forward (copied in by the first segment, out by the last) — without that the residual
`Add`/`RmsNorm` were ineligible and the elementwise kernels bought nothing (27B 1037 -> 1040).
With it: 27B pp512 1037 -> 1067, pp2048 1047 -> 1059; 4B 5242 -> 5811. On the 4B the GEMMs that
still break chains are its Q6_K `ffn_down` / `attn_v` (no fast L0 Q6_K kernel yet: a per-lane
`ushort` load version runs 24 TFLOPS — 104 single-dword loads per superblock; it needs aligned
superblock pairs or a load-time repack).

Where the 27B pp512 goes (before the elementwise chains) (~508 ms per 512 tokens): with the Level Zero launches skipped (wrong
output, timing only) it runs at 2198 t/s, so ~233 ms is everything but the offloaded GEMMs —
attention, DeltaNet, norms, the handoffs, and the ~10% of GEMM work in formats still on Vulkan
(Q6_K, Q3_K, IQ4_NL, IQ3_S at ~45 TFLOPS, ~60 ms); the offloaded GEMMs take ~275 ms (~90 TFLOPS).

**The 2-byte-aligned formats** (`kernels/gemm_kq16.cl`: Q6_K, Q3_K; `gemm_iq4.cl`: IQ4_NL).
Their blocks are 210 / 110 / 18 bytes, so every other block starts 2 bytes off a dword, and IGC
splits a per-lane `ushort` vector load into single-dword messages (Q6_K: 104 per superblock,
24 TFLOPS). The kernels walk blocks in PAIRS, which start dword-aligned when K is a multiple of the
pair (512 / 512 / 64): the odd block's loads are dword vector loads funnel-shifted by 2 bytes, a
compile-time choice per pair half. m=512 5120 -> 17408: Q3_K 0.98 ms (93 TFLOPS), Q6_K 1.25 ms
(73), IQ4_NL 1.36 ms (67, the IQ4_XS table). With them (and default on): 27B pp512 1067 -> 1110,
4B 5811 -> 6278 (the 4B's Q6_K `ffn_down` no longer breaks its chains).

### Long context (2026-11-10, Qwen3.8-27B, 1 rep, llama.cpp `-fa 1`)

| t/s | infr | llama.cpp Vulkan | llama.cpp SYCL |
| --- | --- | --- | --- |
| pp512 @ d8192 | 703 | 375 | 1022 |
| pp512 @ d32768 | 339 | 147 | 894 |
| tg128 @ d8192 | 26.4 | 22.1 | 29.0 |
| tg128 @ d32768 | 20.4 | 19.6 | 27.0 |

(infr before the Q6_K/Q3_K/IQ4_NL kernels.) SYCL's prefill barely moves with depth; ours halves
by 8k and again by 32k: the Vulkan flash attention (`attn_xmx_fa`) at depth is the gap (~7 TFLOPS
effective at 32k). Decode at 32k spends ~14 ms per token on the 2 GB KV read (~150 GB/s; SYCL
~3 ms). The prefill attention change below addresses the first gap; the decode KV read remains.

### Phase 4: DPAS flash-attention prefill

`kernels/attn_fa.cl` adds `attn_fa256`: f16 queries and K/V caches, f32 output, causal or sliding
window, grouped-query attention, custom positive scale (nonpositive means `1/sqrt(256)`), and zero
output for a fully masked row. Its online softmax never materializes a score matrix.

Each SIMD16 subgroup owns eight query rows and **half** their 256-wide output. Two subgroups
recompute the same QK scores and divide the PV output, keeping the output accumulators in
registers: the whole-output version spilled around 140 registers and took 29.45 ms; splitting
the output removed that bottleneck. A 128-thread workgroup covers 32 queries of one head. Q/K/V
use 2D block loads (transposed for K, VNNI-transformed for V), with DPAS for both QK and PV.

On the headless B70 (`03:00.0`), `tests/probe.rs::attn_fa256_probe` measures **16.99 ms / 24.6
effective TFLOPS** at q=512, kv=33280, 24 query heads and 4 KV heads (five launches, median),
against the earlier Vulkan `attn_xmx_fa` measurement of 69 ms. The effective FLOPs count the
mathematical QK+PV work, excluding duplicated QK. The host-f64 reference errors are 3.49e-5 for
GQA causal attention and 6.63e-5 for sliding-window attention.

A 64-key softmax block ran faster in isolation (14.67 ms), but changed the 4B's later greedy
wording on the 1378-token test prompt. Matching Vulkan's 32-key blocks keeps the same f16-P
rescaling boundaries and restores identical output on that check; it is the retained version.

Matched end-to-end comparison (`Vulkan1`, pp512, f16 KV, `INFR_UBATCH=1024`):

| model | depth | Existing L0 GEMMs + elementwise, Vulkan attention | + L0 attention | gain |
| --- | --- | --- | --- | --- |
| Qwen3.8-27B UD-Q4_K_M | 0 | 1114.3 t/s | 1122.9 t/s | 1.01x |
| Qwen3.8-27B UD-Q4_K_M | 32768 | 339.5 t/s | **685.0 t/s** | **2.02x** |
| Qwen3.5-4B Q4_K_M | 0 | 6332.0 t/s | 6335.2 t/s | 1.00x |

Both binaries use the same exportable KV allocations; the adapter's attention offload is the
difference. The short-context row has two timed reps; the 32K row has one. Each benchmark
discards a warm rep at the measured shape, and processes run serially with a 60-second cooldown.
The earlier SYCL 32K result is 894 t/s, so this closes much of the gap without reaching parity.
The 1378-token correctness prompt produces identical 80-token greedy output on both the 4B and
27B with the retained kernel (`INFR_CTX=8192`, `INFR_UBATCH=1024`).

Integration uses the existing bridge and timeline. Device-local KV caches become exportable
allocations when the bridge is wanted; Q and output join the exchange arena. `ze_plan` selects
attention with at least 64 query rows, head dim 256, internal f16 Q, f16 K/V, and exportable caches
that both hold the full live KV range. Ring caches after wrap, quantized KV, attention sinks,
key bias and canvas masks stay on Vulkan. KV overflow placement keeps its existing allocation
path, so those caches also stay on Vulkan.

The adapter regression (`tests/support/ze_attention.rs`, included as a private unit-test module)
checks the selected path against Vulkan and a host-f64 reference: causal GQA, MHA, partial query
and key tiles, a custom scale, fully masked rows, wrapped-cache fallback, and repeated forwards.
It asserts that the bridge is active and that eligible attention actually enters the Level Zero
plan, so a silent Vulkan fallback cannot satisfy the test.

```sh
INFR_ZE_VK=1 cargo test --release -p infr-vulkan --lib ze_attention -- --ignored --nocapture --test-threads=1
INFR_ZE_PCI=03:00.0 cargo test --release -p infr-ze --test probe attn_fa256_probe -- --ignored --nocapture --test-threads=1
```

### Next

- Further prefill attention tuning and the decode KV read — the remaining long-context gap.
- Default-context placement: the earlier 27B run clamped its unpinned context to 50640 and
  reduced the prefill chunk to 128 rows to fit it, below the L0 GEMMs' 256-row floor. The explicit
  context/chunk settings above avoid that policy tradeoff; it is unchanged by this slice.
- IQ4_XS: fewer SLM gathers — e.g. a byte-indexed table (k-pairs `(j, j+16)`, one index per byte)
  with the producer writing A in the matching k order.
- IQ3_S, Q8_0, Q2_K (not in the 27B/4B's GEMM paths).
